
/*--------------------------------------------------------------------*/
/*--- FeatRed: Based on Debgrind, with reduced features. fr_main.c ---*/
/*--------------------------------------------------------------------*/

/* DSK -- Modifying Nulgrind to make Debgrind, to be lighter weight
   than the tool based on lackey. */

/*
   This file is part of Nulgrind, the minimal Valgrind tool,
   which does no instrumentation or analysis.

   Copyright (C) 2002-2017 Nicholas Nethercote
      njn@valgrind.org

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307, USA.

   The GNU General Public License is contained in the file COPYING.
*/

#include "pub_tool_basics.h"
#include "pub_tool_tooliface.h"
#include "pub_tool_libcassert.h"
#include "pub_tool_libcprint.h"
#include "pub_tool_debuginfo.h"
#include "pub_tool_libcbase.h"
#include "pub_tool_libcproc.h"
#include "pub_tool_options.h"
#include "pub_tool_machine.h"     // VG_(fnptr_to_fnentry)
#include "pub_tool_vki.h"
#include "pub_tool_libcfile.h"
#include "pub_tool_xtree.h"
/*------------------------------------------------------------*/
/*--- Command line options                                 ---*/
/*------------------------------------------------------------*/
static const HChar* clo_output_file = "output.out";
static UInt clo_ins_dump_interval = 0;
static UInt clo_ms_dump_interval = 0;

static Bool fr_process_cmd_line_option(const HChar* arg)
{
  if VG_STR_CLO(arg, "--output-file",            clo_output_file) {}
  else if VG_INT_CLO(arg, "--ins-dump-interval", clo_ins_dump_interval) {}
  else if VG_INT_CLO(arg, "--ms-dump-interval",  clo_ms_dump_interval) {}
  else
    return False;
  tl_assert(clo_output_file);
  tl_assert(clo_output_file[0]);
  return True;
}

static void fr_print_usage(void)
{
   VG_(printf)(
"    --output-file=<name>           Specify a file for output.\n"
"    --ins-dump-interval=<int>      Dump a summary every X instructions.\n"
   );
}

static void fr_print_debug_usage(void)
{
   VG_(printf)(
"    (none)\n"
   );
}
/*------------------------------------------------------------*/
/*--- Globals                                              ---*/
/*------------------------------------------------------------*/
static Time start_time_ms = 0;
static ULong sb_enter = 0;
static ULong sb_exit = 0;
static ULong ins_count = 0;
static ULong max_ins_addr = 0;
static ULong min_ins_addr = 0;

static ULong store_count = 0;
static ULong wrtmp_count = 0;
static ULong exit_count = 0;

static Addr cur_ins_addr = 0;
static UInt cur_ins_len = 0;

typedef
  IRExpr
  IRAtom;

VgFile* output_file;
static ULong next_dump_ins_count = 0;

/*------------------------------------------------------------*/
/*--- Instrumentation                                      ---*/
/*------------------------------------------------------------*/
static void fr_post_clo_init(void)
{
  output_file = VG_(fopen)(clo_output_file,
				   VKI_O_WRONLY|VKI_O_CREAT|VKI_O_TRUNC,
				   VKI_S_IRUSR|VKI_S_IWUSR);
}

static void fr_dump_stats(ULong ins_tick_count) {
  ULong ins_addr_diff;
  VG_(fprintf)(output_file, "%llu\n", ins_tick_count);
  VG_(fprintf)(output_file, "SBEnter %llu\n", sb_enter);
  VG_(fprintf)(output_file, "SBExit %llu\n", sb_exit);
  VG_(fprintf)(output_file, "InsCount %llu\n", ins_count);
  VG_(fprintf)(output_file, "MaxInsAddr %llu\n", max_ins_addr);
  VG_(fprintf)(output_file, "MinInsAddr %llu\n", min_ins_addr);
  ins_addr_diff = max_ins_addr - min_ins_addr;
  VG_(fprintf)(output_file, "InsAddrDiff %llu\n", ins_addr_diff);
  VG_(fprintf)(output_file, "StoreCount %llu\n", store_count);
  VG_(fprintf)(output_file, "WrTmpCount %llu\n", wrtmp_count);
  VG_(fprintf)(output_file, "ExitCount %llu\n", exit_count);
  VG_(fprintf)(output_file, "\n");
}


static
IRSB* fr_instrument ( VgCallbackClosure* closure,
                      IRSB* bb,
                      const VexGuestLayout* layout,
                      const VexGuestExtents* vge,
                      const VexArchInfo* archinfo_host,
                      IRType gWordTy, IRType hWordTy )
{
  Addr iaddr = 0;
  Int  i;
  UInt ilen = 0;
  //IRTypeEnv* tyenv = bb->tyenv;
  //Addr dataAddr = 0;
  //IRExpr* data;
  //IRAtom* addr_expr;
  Bool dump = False;

  // If we are exactly at a dump interval
  dump = ((clo_ins_dump_interval != 0) &&
	  (ins_count % clo_ins_dump_interval == 0));
  // If we've recently missed a dump interval

  if (next_dump_ins_count <= ins_count && next_dump_ins_count != 0) {
    dump = True;
  }
  if (dump) {
    fr_dump_stats(next_dump_ins_count);
    if (next_dump_ins_count != 0) {
      next_dump_ins_count = (((ins_count / clo_ins_dump_interval) + 1) *
			     clo_ins_dump_interval);
    }
    else {next_dump_ins_count = clo_ins_dump_interval;}
  }

  sb_enter++;

  i = 0;
  while (i < bb->stmts_used && bb->stmts[i]->tag != Ist_IMark) {
    i++;
  }
  for (; i < bb->stmts_used; i++) {
    IRStmt* st = bb->stmts[i];
    if (!st || st->tag == Ist_NoOp) continue;
    switch (st->tag) {
    case Ist_NoOp:
    case Ist_AbiHint:
    case Ist_Put:
    case Ist_PutI:
    case Ist_MBE:
    case Ist_LoadG:
    case Ist_Dirty:
    case Ist_CAS:
    case Ist_LLSC:
    case Ist_StoreG:
      break;
    case Ist_IMark:
      /*  The "IMark" is an IR statement that doesn't represent actual code.
   Instead it indicates the address and length of the original
   instruction. (See VEX/pub/libvex_ir.h, which has useful documentation.)
       */
      ins_count++;

      iaddr = st->Ist.IMark.addr;
      ilen = st->Ist.IMark.len;
      if (iaddr > max_ins_addr || max_ins_addr == 0) {
	max_ins_addr = iaddr;
      }
      if (iaddr < min_ins_addr || min_ins_addr == 0) {
	min_ins_addr = iaddr;
      }
      cur_ins_addr = iaddr;
      cur_ins_len = ilen;

      break;
    case Ist_WrTmp:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      wrtmp_count++;
      break;
    case Ist_Store:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);

      //IRExpr* data = st->Ist.Store.data;
      //IRType type = typeOfIRExpr(tyenv, data);
      //dataAddr = (ULong) st->Ist.Store.data;
      store_count++;

      break;
    case Ist_Exit:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      exit_count++;
      break;
    default:
      tl_assert(0);
    }
  }
  sb_exit++;

  return bb;
}


static void fr_fini(Int exitcode)
{
  Time end_time_ms = VG_(read_millisecond_timer)();
  VG_(umsg)("Elapsed tool time: %lld\n", end_time_ms - start_time_ms);


  fr_dump_stats(ins_count);
  VG_(fclose)(output_file);
}

static void fr_pre_clo_init(void)
{
  start_time_ms = VG_(read_millisecond_timer)();
  VG_(details_name)            ("FeatRed");
  VG_(details_version)         (NULL);
  VG_(details_description)     ("based on the minimal Valgrind tool");
  VG_(details_copyright_author)("Copyright (C) 2002-2017, and GNU GPL'd, by Nicholas Nethercote, modified by Deby Katz 2017-2019.");
  VG_(details_bug_reports_to)  (VG_BUGS_TO);

  VG_(details_avg_translation_sizeB) ( 275 );

  VG_(basic_tool_funcs)        (fr_post_clo_init,
				fr_instrument,
				fr_fini);

  VG_(needs_command_line_options)(fr_process_cmd_line_option,
				  fr_print_usage,
				  fr_print_debug_usage);
}

VG_DETERMINE_INTERFACE_VERSION(fr_pre_clo_init)

/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
