
/*--------------------------------------------------------------------*/
/*--- Debgrind: Based on the minimal Valgrind tool.      dg_main.c ---*/
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
static ULong clo_ins_dump_interval = 0;
static ULong clo_ms_dump_interval = 0;

static Bool dg_process_cmd_line_option(const HChar* arg)
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

static void dg_print_usage(void)
{
   VG_(printf)(
"    --output-file=<name>           Specify a file for output.\n"
"    --ins-dump-interval=<int>      Dump a summary every X instructions.\n"
   );
}

static void dg_print_debug_usage(void)
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
static ULong ir_stmts = 0;
static ULong max_ins_addr = 0;
static ULong min_ins_addr = 0;
static ULong max_loadg_addr = 0;
static ULong min_loadg_addr = 0;
static ULong max_store_addr = 0;
static ULong min_store_addr = 0;
static ULong max_storeg_addr = 0;
static ULong min_storeg_addr = 0;
static ULong min_wrtmp_addr = 0;
static ULong max_wrtmp_addr = 0;

static ULong store_count = 0;
static ULong wrtmp_count = 0;
static ULong storeg_count = 0;
static ULong loadg_count = 0;
static ULong dirty_count = 0;
static ULong cas_count = 0;
static ULong llsc_count = 0;
static ULong exit_count = 0;


static Addr cur_ins_addr = 0;
static UInt cur_ins_len = 0;

typedef
  IRExpr
  IRAtom;

VgFile* output_file;

/*------------------------------------------------------------*/
/*--- Instrumentation                                      ---*/
/*------------------------------------------------------------*/
static void dg_post_clo_init(void)
{
  output_file = VG_(fopen)(clo_output_file,
				   VKI_O_WRONLY|VKI_O_CREAT|VKI_O_TRUNC,
				   VKI_S_IRUSR|VKI_S_IWUSR);
}

static void dg_dump_stats(void) {
  ULong ins_addr_diff;
  ULong load_addr_diff;
  ULong store_addr_diff;
  ULong wrtmp_addr_diff;
  ULong storeg_addr_diff;
  VG_(fprintf)(output_file, "SBEnter %llu\n", sb_enter);
  VG_(fprintf)(output_file, "SBExit %llu\n", sb_exit);
  VG_(fprintf)(output_file, "InsCount %llu\n", ins_count);
  VG_(fprintf)(output_file, "MaxInsAddr %llu\n", max_ins_addr);
  VG_(fprintf)(output_file, "MinInsAddr %llu\n", min_ins_addr);
  ins_addr_diff = max_ins_addr - min_ins_addr;
  VG_(fprintf)(output_file, "InsAddrDiff %llu\n", ins_addr_diff);
  VG_(fprintf)(output_file, "MaxLoadAddr %llu\n", max_loadg_addr);
  VG_(fprintf)(output_file, "MinLoadAddr %llu\n", min_loadg_addr);
  load_addr_diff = max_loadg_addr - min_loadg_addr;
  VG_(fprintf)(output_file, "LoadAddrDiff %llu\n", load_addr_diff);
  VG_(fprintf)(output_file, "LoadCount %llu\n", loadg_count);
  VG_(fprintf)(output_file, "MaxStoreAddr %llu\n", max_store_addr);
  VG_(fprintf)(output_file, "MinStoreAddr %llu\n", min_store_addr);
  store_addr_diff = max_store_addr - min_store_addr;
  VG_(fprintf)(output_file, "StoreAddrDiff %llu\n", store_addr_diff);
  VG_(fprintf)(output_file, "StoreCount %llu\n", store_count);
  VG_(fprintf)(output_file, "MinWrTmpAddr %llu\n", min_wrtmp_addr);
  VG_(fprintf)(output_file, "MaxWrTmpAddr %llu\n", max_wrtmp_addr);
  wrtmp_addr_diff = max_wrtmp_addr - min_wrtmp_addr;
  VG_(fprintf)(output_file, "WrTmpDiff %llu\n", wrtmp_addr_diff);
  VG_(fprintf)(output_file, "WrTmpCount %llu\n", wrtmp_count);
  VG_(fprintf)(output_file, "MaxStoreGAddr %llu\n", max_storeg_addr);
  VG_(fprintf)(output_file, "MinStoreGAddr %llu\n", min_storeg_addr);
  storeg_addr_diff = max_storeg_addr - min_storeg_addr;
  VG_(fprintf)(output_file, "StoreGAddrDiff %llu\n", storeg_addr_diff);
  VG_(fprintf)(output_file, "StoreGCount %llu\n", storeg_count);
  VG_(fprintf)(output_file, "DirtyCount %llu\n", dirty_count);
  VG_(fprintf)(output_file, "CASCount %llu\n", cas_count);
  VG_(fprintf)(output_file, "LLSCCount %llu\n", llsc_count);
  VG_(fprintf)(output_file, "ExitCount %llu\n", exit_count);
  VG_(fprintf)(output_file, "\n");
}


static
IRSB* dg_instrument ( VgCallbackClosure* closure,
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
  Addr dataAddr = 0;
  IRExpr* data;
  IRAtom* addr_expr;
  Bool dump = False;

  sb_enter++;

  i = 0;
  while (i < bb->stmts_used && bb->stmts[i]->tag != Ist_IMark) {
    i++;
  }
  for (; i < bb->stmts_used; i++) {
    IRStmt* st = bb->stmts[i];
    if (!st || st->tag == Ist_NoOp) continue;
    ir_stmts++;

    switch (st->tag) {
    case Ist_NoOp:
    case Ist_AbiHint:
    case Ist_Put:
    case Ist_PutI:
    case Ist_MBE:
      break;
    case Ist_IMark:
      /*  The "IMark" is an IR statement that doesn't represent actual code.
   Instead it indicates the address and length of the original
   instruction. (See VEX/pub/libvex_ir.h, which has useful documentation.)
       */
      ins_count++;
      dump = ((clo_ins_dump_interval != 0) &&
	      (ins_count % clo_ins_dump_interval == 0));
      if (dump) {
	dg_dump_stats();
      }

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

      data = st->Ist.WrTmp.data;
      if (data->tag == Iex_Load) {
	addr_expr = data->Iex.Load.addr;

	if ((ULong)addr_expr > max_wrtmp_addr || max_wrtmp_addr == 0) {
	  max_wrtmp_addr = (ULong)addr_expr;
	}
	if ((ULong)addr_expr < min_wrtmp_addr || min_wrtmp_addr == 0) {
	  min_wrtmp_addr = (ULong)addr_expr;
	}
      }
      break;
    case Ist_Store:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);

      //IRExpr* data = st->Ist.Store.data;
      //IRType type = typeOfIRExpr(tyenv, data);
      dataAddr = (ULong) st->Ist.Store.data;
      if (dataAddr < min_store_addr || min_store_addr == 0) {
	min_store_addr = dataAddr;
      }
      if (dataAddr > max_store_addr || max_store_addr == 0) {
	max_store_addr = dataAddr;
      }
      store_count++;

      break;
    case Ist_StoreG:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      dataAddr = (ULong) st->Ist.StoreG.details->addr;
      if (dataAddr < min_storeg_addr || min_storeg_addr == 0) {
	min_storeg_addr = dataAddr;
      }
      if (dataAddr > max_storeg_addr || max_storeg_addr == 0) {
	max_storeg_addr = dataAddr;
      }
      storeg_count++;
      break;
    case Ist_LoadG:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      dataAddr = (ULong) st->Ist.LoadG.details->addr;
      if (dataAddr < min_loadg_addr || min_loadg_addr == 0) {
	min_loadg_addr = dataAddr;
      }
      if (dataAddr > max_loadg_addr || max_loadg_addr == 0) {
	max_loadg_addr = dataAddr;
      }
      loadg_count++;
      break;
    case Ist_Dirty:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      dirty_count++;
      break;
    case Ist_CAS:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      cas_count++;
      break;
    case Ist_LLSC:
      //tl_assert(cur_ins_addr != 0);
      //tl_assert(cur_ins_len != 0);
      llsc_count++;
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


static void dg_fini(Int exitcode)
{
  Time end_time_ms = VG_(read_millisecond_timer)();
  VG_(umsg)("Elapsed tool time: %lld\n", end_time_ms - start_time_ms);


  dg_dump_stats();
  VG_(fclose)(output_file);
}

static void dg_pre_clo_init(void)
{
  start_time_ms = VG_(read_millisecond_timer)();
  VG_(details_name)            ("Debgrind");
  VG_(details_version)         (NULL);
  VG_(details_description)     ("based on the minimal Valgrind tool");
  VG_(details_copyright_author)("Copyright (C) 2002-2017, and GNU GPL'd, by Nicholas Nethercote, modified by Deby Katz 2017-2019.");
  VG_(details_bug_reports_to)  (VG_BUGS_TO);

  VG_(details_avg_translation_sizeB) ( 275 );

  VG_(basic_tool_funcs)        (dg_post_clo_init,
				dg_instrument,
				dg_fini);

  VG_(needs_command_line_options)(dg_process_cmd_line_option,
				  dg_print_usage,
				  dg_print_debug_usage);
}

VG_DETERMINE_INTERFACE_VERSION(dg_pre_clo_init)

/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
