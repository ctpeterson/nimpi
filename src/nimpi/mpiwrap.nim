#[
  NiMPI: https://github.com/ctpeterson/nimpi
  Source: src/nimpi/mpiwrap.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Acknowledgments:
    - NimMPI by Michalina Kotwica (Udiknedormin)
      https://github.com/Udiknedormin/NimMPI — MIT License
      Copyright (c) 2016 M. Kotwica
    - QEX (Quantum EXpressions) by James Osborn et al.
      https://github.com/jcosborn/qex — MIT License
      Copyright (c) 2015 James Osborn

  MIT License

  Copyright (c) 2026 Curtis Taylor Peterson

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
]#

import std/[macros]

{.pragma: mpi, importc, header: "mpi.h".}

#[ macros for slick definitions of types and constants ]#

macro mpiTypes(definition, idents: untyped): untyped =
  result = newStmtList()
  for ident in idents: result.add newCall(definition, ident)

macro mpiVariables(definition, idents: untyped): untyped =
  result = newStmtList()
  for ident in idents:
    if ident[1].kind == nnkStmtList: 
      result.add newCall(definition, ident[0], ident[1][0])
    else: result.add newCall(definition, ident[0], ident[1])

template typeDefinition(ident: untyped) {.dirty.} =
  type ident* {.mpi.} = object

template constantDefinition(ident: untyped, identType: untyped) {.dirty.} =
  var ident* {.mpi.}: identType

#[ type & constant definitions ]#

type MPI_Offset* {.mpi.} = cint

type MPI_Status* {.importc: "MPI_Status", mpi.} = object
  ## MPI status returned by receive and probe operations.
  ## MPI_SOURCE, MPI_TAG, and MPI_ERROR are guaranteed by the MPI standard.
  MPI_SOURCE*: cint
  MPI_TAG*: cint
  MPI_ERROR*: cint

mpiTypes typeDefinition:
  MPI_Aint
  MPI_Count
  MPI_Comm
  MPI_Datatype
  MPI_Errhandler
  MPI_File
  MPI_Group
  MPI_Info
  MPI_Op
  MPI_Request
  MPI_Message
  MPI_Win
  MPI_Fint

mpiVariables constantDefinition:
  # communicator constants
  MPI_COMM_WORLD: MPI_Comm
  MPI_COMM_SELF: MPI_Comm
  MPI_COMM_NULL: MPI_Comm

  # group constants
  MPI_GROUP_NULL: MPI_Group
  MPI_GROUP_EMPTY: MPI_Group

  # special rank values
  MPI_ANY_SOURCE: cint
  MPI_PROC_NULL: cint
  MPI_ROOT: cint
  MPI_UNDEFINED: cint

  # reduction operations
  MPI_MAX: MPI_Op
  MPI_MIN: MPI_Op
  MPI_SUM: MPI_Op
  MPI_PROD: MPI_Op
  MPI_LAND: MPI_Op
  MPI_BAND: MPI_Op
  MPI_LOR: MPI_Op
  MPI_BOR: MPI_Op
  MPI_LXOR: MPI_Op
  MPI_BXOR: MPI_Op
  MPI_MAXLOC: MPI_Op
  MPI_MINLOC: MPI_Op
  MPI_REPLACE: MPI_Op
  MPI_NO_OP: MPI_Op
  MPI_OP_NULL: MPI_Op

  # datatypes
  MPI_DATATYPE_NULL: MPI_Datatype
  MPI_BYTE: MPI_Datatype
  MPI_PACKED: MPI_Datatype
  MPI_CHAR: MPI_Datatype
  MPI_SHORT: MPI_Datatype
  MPI_INT: MPI_Datatype
  MPI_LONG: MPI_Datatype
  MPI_FLOAT: MPI_Datatype
  MPI_DOUBLE: MPI_Datatype
  MPI_LONG_DOUBLE: MPI_Datatype
  MPI_UNSIGNED_CHAR: MPI_Datatype
  MPI_SIGNED_CHAR: MPI_Datatype
  MPI_UNSIGNED_SHORT: MPI_Datatype
  MPI_UNSIGNED: MPI_Datatype
  MPI_UNSIGNED_LONG: MPI_Datatype
  MPI_UNSIGNED_LONG_LONG: MPI_Datatype
  MPI_LONG_LONG_INT: MPI_Datatype
  MPI_INT8_T: MPI_Datatype
  MPI_UINT8_T: MPI_Datatype
  MPI_INT16_T: MPI_Datatype
  MPI_UINT16_T: MPI_Datatype
  MPI_INT32_T: MPI_Datatype
  MPI_UINT32_T: MPI_Datatype
  MPI_INT64_T: MPI_Datatype
  MPI_UINT64_T: MPI_Datatype
  MPI_C_BOOL: MPI_Datatype
  MPI_C_FLOAT_COMPLEX: MPI_Datatype
  MPI_C_DOUBLE_COMPLEX: MPI_Datatype
  MPI_FLOAT_INT: MPI_Datatype
  MPI_DOUBLE_INT: MPI_Datatype
  MPI_LONG_INT: MPI_Datatype
  MPI_SHORT_INT: MPI_Datatype
  MPI_2INT: MPI_Datatype

  # Request / status
  MPI_REQUEST_NULL: MPI_Request
  MPI_MESSAGE_NULL: MPI_Message
  MPI_MESSAGE_NO_PROC: MPI_Message
  MPI_STATUS_IGNORE: ptr MPI_Status
  MPI_STATUSES_IGNORE: ptr MPI_Status

  # Info
  MPI_INFO_NULL: MPI_Info

  # Errhandler
  MPI_ERRHANDLER_NULL: MPI_Errhandler
  MPI_ERRORS_ARE_FATAL: MPI_Errhandler
  MPI_ERRORS_RETURN: MPI_Errhandler

  # Misc
  MPI_IN_PLACE: pointer
  MPI_BOTTOM: pointer
  MPI_BSEND_OVERHEAD: cint
  MPI_MAX_PROCESSOR_NAME: cint
  MPI_MAX_ERROR_STRING: cint
  MPI_MAX_OBJECT_NAME: cint
  MPI_MAX_LIBRARY_VERSION_STRING: cint

  # Error codes
  MPI_SUCCESS: cint
  MPI_ERR_BUFFER: cint
  MPI_ERR_COUNT: cint
  MPI_ERR_TYPE: cint
  MPI_ERR_TAG: cint
  MPI_ERR_COMM: cint
  MPI_ERR_RANK: cint
  MPI_ERR_REQUEST: cint
  MPI_ERR_ROOT: cint
  MPI_ERR_GROUP: cint
  MPI_ERR_OP: cint
  MPI_ERR_TOPOLOGY: cint
  MPI_ERR_DIMS: cint
  MPI_ERR_ARG: cint
  MPI_ERR_UNKNOWN: cint
  MPI_ERR_TRUNCATE: cint
  MPI_ERR_OTHER: cint
  MPI_ERR_INTERN: cint
  MPI_ERR_IN_STATUS: cint
  MPI_ERR_PENDING: cint
  MPI_ERR_LASTCODE: cint

  # Thread support levels
  MPI_THREAD_SINGLE: cint
  MPI_THREAD_FUNNELED: cint
  MPI_THREAD_SERIALIZED: cint
  MPI_THREAD_MULTIPLE: cint

  # Topology types
  MPI_CART: cint
  MPI_GRAPH: cint
  MPI_DIST_GRAPH: cint

  # Communicator comparison results
  MPI_IDENT: cint
  MPI_CONGRUENT: cint
  MPI_SIMILAR: cint
  MPI_UNEQUAL: cint

  # File constants
  MPI_FILE_NULL: MPI_File
  MPI_MODE_RDONLY: cint
  MPI_MODE_RDWR: cint
  MPI_MODE_WRONLY: cint
  MPI_MODE_CREATE: cint
  MPI_MODE_APPEND: cint

#[ initialization/finalization ]#

proc MPI_Init*(argc: ptr cint, argv: ptr cstringArray): cint {.cdecl, mpi.}

proc MPI_Initialized*(flag: ptr cint): cint {.cdecl, mpi.}

proc MPI_Finalize*: cint {.cdecl, mpi.}

proc MPI_Finalized*(flag: ptr cint): cint {.cdecl, mpi.}

#[ error handling ]#

proc MPI_Error_string*(
  code: cint, 
  str: cstring, 
  resultlen: ptr cint
): cint {.cdecl, mpi.}

#[ timers ]#

proc MPI_Wtime*: cdouble {.cdecl, mpi.}

proc MPI_Wtick*: cdouble {.cdecl, mpi.}

#[ MPI status ]#

proc MPI_Get_count*(
  status: ptr MPI_Status,
  datatype: MPI_Datatype,
  count: ptr cint
): cint {.cdecl, mpi.}

#[ MPI group ]#

proc MPI_Group_incl*(
  group: MPI_Group, 
  n: cint, 
  ranks: ptr cint, 
  newgroup: ptr MPI_Group
): cint {.cdecl, mpi.}

proc MPI_Group_rank*(group: MPI_Group, rank: ptr cint): cint {.cdecl, mpi.}

proc MPI_Group_size*(group: MPI_Group, size: ptr cint): cint {.cdecl, mpi.}

proc MPI_Group_free*(group: ptr MPI_Group): cint {.cdecl, mpi.}

proc MPI_Group_union*(
  group1: MPI_Group, 
  group2: MPI_Group, 
  newgroup: ptr MPI_Group
): cint {.cdecl, mpi.}

proc MPI_Group_intersection*(
  group1: MPI_Group, 
  group2: MPI_Group, 
  newgroup: ptr MPI_Group
): cint {.cdecl, mpi.}

proc MPI_Group_difference*(
  group1: MPI_Group, 
  group2: MPI_Group, 
  newgroup: ptr MPI_Group
): cint {.cdecl, mpi.}

#[ MPI communicator ]#

proc MPI_Abort*(comm: MPI_Comm, errorcode: cint): cint {.cdecl, mpi.}

proc MPI_Comm_rank*(comm: MPI_Comm, rank: ptr cint): cint {.cdecl, mpi.}

proc MPI_Comm_size*(comm: MPI_Comm, size: ptr cint): cint {.cdecl, mpi.}

proc MPI_Comm_dup*(comm: MPI_Comm, newcomm: ptr MPI_Comm): cint {.cdecl, mpi.}

proc MPI_Comm_split*(
  comm: MPI_Comm, 
  color: cint, 
  key: cint, 
  newcomm: ptr MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Comm_free*(comm: ptr MPI_Comm): cint {.cdecl, mpi.}

proc MPI_Comm_create*(
  comm: MPI_Comm, 
  group: MPI_Group, 
  newcomm: ptr MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Comm_compare*(
  comm1: MPI_Comm, 
  comm2: MPI_Comm, 
  result: ptr cint
): cint {.cdecl, mpi.}

proc MPI_Comm_group*(comm: MPI_Comm, group: ptr MPI_Group): cint {.cdecl, mpi.}

proc MPI_Barrier*(comm: MPI_Comm): cint {.cdecl, mpi.}

proc MPI_Probe*(
  source: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

proc MPI_Iprobe*(
  source: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  flag: ptr cint, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

#[ MPI request ]#

proc MPI_Wait*(request: ptr MPI_Request, status: ptr MPI_Status): cint {.cdecl, mpi.}

proc MPI_Test*(
  request: ptr MPI_Request, 
  flag: ptr cint, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

proc MPI_Waitall*(
  count: cint, 
  array_of_requests: ptr MPI_Request, 
  array_of_statuses: ptr MPI_Status
): cint {.cdecl, mpi.}

proc MPI_Testall*(
  count: cint, 
  array_of_requests: ptr MPI_Request, 
  flag: ptr cint, 
  array_of_statuses: ptr MPI_Status
): cint {.cdecl, mpi.}

#[ MPI send/recv ]#

proc MPI_Send*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Ssend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Bsend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Rsend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Isend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  request: ptr MPI_Request
): cint {.cdecl, mpi.}

proc MPI_Issend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  request: ptr MPI_Request
): cint {.cdecl, mpi.}

proc MPI_Ibsend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  request: ptr MPI_Request
): cint {.cdecl, mpi.}

proc MPI_Irsend*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  request: ptr MPI_Request
): cint {.cdecl, mpi.}

proc MPI_Recv*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  source: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

proc MPI_Irecv*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  source: cint, 
  tag: cint, 
  comm: MPI_Comm, 
  request: ptr MPI_Request
): cint {.cdecl, mpi.}

proc MPI_Sendrecv*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  dest: cint, 
  sendtag: cint, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  source: cint, 
  recvtag: cint, 
  comm: MPI_Comm, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

proc MPI_Sendrecv_replace*(
  buf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  dest: cint, 
  sendtag: cint, 
  source: cint, 
  recvtag: cint, 
  comm: MPI_Comm, 
  status: ptr MPI_Status
): cint {.cdecl, mpi.}

#[ collective operations ]#

proc MPI_Bcast*(
  buffer: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Gather*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Allgather*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Gatherv*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcounts: ptr cint, 
  displs: ptr cint, 
  recvtype: MPI_Datatype, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Scatter*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Scatterv*(
  sendbuf: pointer, 
  sendcounts: ptr cint, 
  displs: ptr cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Alltoall*(
  sendbuf: pointer, 
  sendcount: cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcount: cint, 
  recvtype: MPI_Datatype, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Alltoallv*(
  sendbuf: pointer, 
  sendcounts: ptr cint, 
  sdispls: ptr cint, 
  sendtype: MPI_Datatype, 
  recvbuf: pointer, 
  recvcounts: ptr cint, 
  rdispls: ptr cint, 
  recvtype: MPI_Datatype, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

#[
proc MPI_Alltoallw*(
  sendbuf: pointer, 
  sendcounts: ptr cint, 
  sdispls: ptr cint, 
  sendtypes: ptr MPI_Datatype, 
  recvbuf: pointer, 
  recvcounts: ptr cint, 
  rdispls: ptr cint, 
  recvtypes: ptr MPI_Datatype, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}
]#

#[ reduction operations (also collective, of course) ]#

proc MPI_Reduce*(
  sendbuf: pointer, 
  recvbuf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  op: MPI_Op, 
  root: cint, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Allreduce*(
  sendbuf: pointer, 
  recvbuf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  op: MPI_Op, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Scan*(
  sendbuf: pointer, 
  recvbuf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  op: MPI_Op, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

proc MPI_Exscan*(
  sendbuf: pointer, 
  recvbuf: pointer, 
  count: cint, 
  datatype: MPI_Datatype, 
  op: MPI_Op, 
  comm: MPI_Comm
): cint {.cdecl, mpi.}

