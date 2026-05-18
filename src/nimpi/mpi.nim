#[
  NiMPI: https://github.com/ctpeterson/nimpi
  Source: src/nimpi/mpi.nim
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

## NiMPI – MPI Wrapper for Nim
##
## High-level, idiomatic Nim bindings for Message Passing Interface (MPI).
## This is a work in progress and not ready for production use. API is subject
## to change without deprecation.
##
## Hello World
## ===========
## 
## The simplest way to use NiMPI is to annotate a block of code with the `mpi` pragma, 
## which wraps the code in an MPI initialization/finalization block:
##
## ```nim
## import nimpi
##
## mpi:
##   let rank = WorldCommunicator.myRank # rank of the calling process
##   let size = WorldCommunicator.size   # number of ranks in the communicator
##   echo "Hello, world! From: ", rank, "/", size
## ```
## 
## Alternatively, one can annotate a procedure with the `mpi` pragma, which wraps
## the body of the procedure in an MPI initialization/finalization block:
## 
## ```nim
## import nimpi
##
## proc main() {.mpi.} =
##   let rank = WorldCommunicator.myRank # rank of the calling process
##   let size = WorldCommunicator.size   # number of ranks in the communicator
##   echo "Hello, world! From: ", rank, "/", size
## main()
## ```
## 
## Both modes of initialization and finalization are functionally equivalent the 
## most verbose means of initializing and finalizing MPI with NiMPI:
## 
## ```nim
## import nimpi
##
## mpiInit()
## let rank = WorldCommunicator.myRank # rank of the calling process
## let size = WorldCommunicator.size   # number of ranks in the communicator
## echo "Hello, world! From: ", rank, "/", size
## mpiFinalize()
## ```

import std/[macros]
import mpiwrap

#[ MPI types ]#

type MpiError* = object of CatchableError
  ## Represents an error that occurs during MPI operations. Contains an error code
  ## that can be used to identify the specific MPI error that occurred. Paired with
  ## the `mpiCheck` procedure, which raises an `MpiError` when an MPI function returns
  ## an error code.
  ## 
  ## Attributes:
  ##  - `code`: The MPI error code associated with the error. 
  code*: cint

type
  SendType* = enum
    ## Represents the type of send operation for point-to-point communication.
    ##
    ## Values:
    ## - `StandardSend`: corresponds to `MPI_Send`
    ## - `BufferedSend`: corresponds to `MPI_Bsend`
    ## - `SynchronousSend`: corresponds to `MPI_Ssend`
    ## - `ReadySend`: corresponds to `MPI_Rsend`
    StandardSend,
    BufferedSend,
    SynchronousSend,
    ReadySend

type 
  MpiCommunicator* = object
    ## Represents an MPI communicator, which is a group of processes that can 
    ## communicate with each other. MPI communicators are fundamental to MPI 
    ## programming, as they define the context for communication operations.
    ## 
    ## Attributes:
    ## - `comm`: The underlying MPI_Comm handle that represents the communicator.
    comm*: MPI_Comm

  MpiRequest* = object
    ## Represents an MPI request, which is used for non-blocking communication operations. 
    ## MPI requests allow for asynchronous communication, enabling processes to perform 
    ## other work while waiting for communication to complete.
    ##
    ## Attributes:
    ## - `request`: The underlying MPI_Request handle that represents the request.
    request*: mpiwrap.MPI_Request
  
  MpiGroup* = object
    ## Represents an MPI group, which is an ordered set of processes. MPI groups
    ## are used to define the membership of communicators and to specify subsets
    ## of processes for communication operations.
    ##
    ## Attributes:
    ## - `group`: The underlying MPI_Group handle that represents the group.
    group*: mpiwrap.MPI_Group

  MpiStatus* = object
    ## Represents the status of an MPI operation, which contains information about
    ## the source, tag, and error code of a completed communication operation. MPI
    ## statuses are used in receive operations to determine the details of the 
    ## received message.
    ##
    ## Attributes:
    ## - `status`: The underlying MPI_Status handle that represents the status.
    status*: mpiwrap.MPI_Status

type
  MpiOperation* = enum
    ## Represents an MPI reduction operation for use in `reduce` and related calls.
    ##
    ## Values:
    ## - `ReduceSum`: corresponds to `MPI_SUM`
    ## - `ReduceProduct`: corresponds to `MPI_PROD`
    ## - `ReduceMaximum`: corresponds to `MPI_MAX`
    ## - `ReduceMinimum`: corresponds to `MPI_MIN`
    ## - `ReduceLogicalAnd`: corresponds to `MPI_LAND`
    ## - `ReduceBitwiseAnd`: corresponds to `MPI_BAND`
    ## - `ReduceLogicalOr`: corresponds to `MPI_LOR`
    ## - `ReduceBitwiseOr`: corresponds to `MPI_BOR`
    ## - `ReduceLogicalXor`: corresponds to `MPI_LXOR`
    ## - `ReduceBitwiseXor`: corresponds to `MPI_BXOR`
    ## - `ReduceMaxLoc`: corresponds to `MPI_MAXLOC`
    ## - `ReduceMinLoc`: corresponds to `MPI_MINLOC`
    ## - `ReduceReplace`: corresponds to `MPI_REPLACE`
    ## - `ReduceNoOp`: corresponds to `MPI_NO_OP`
    ReduceSum,
    ReduceProduct,
    ReduceMaximum,
    ReduceMinimum,
    ReduceLogicalAnd,
    ReduceBitwiseAnd,
    ReduceLogicalOr,
    ReduceBitwiseOr,
    ReduceLogicalXor,
    ReduceBitwiseXor,
    ReduceMaxLoc,
    ReduceMinLoc,
    ReduceReplace,
    ReduceNoOp

#[ global variables ]#

let
  WorldCommunicator* = MpiCommunicator(comm: MPI_COMM_WORLD)
    ## The default communicator that includes all processes in the MPI program.
    
  SelfCommunicator* = MpiCommunicator(comm: MPI_COMM_SELF)
    ## A communicator that includes only the calling process. 
  
  NullCommunicator* = MpiCommunicator(comm: MPI_COMM_NULL)
    ## A null communicator that represents an invalid communicator.

let
  NullGroup* = MpiGroup(group: MPI_GROUP_NULL)
    ## A null group that represents an invalid group.

  EmptyGroup* = MpiGroup(group: MPI_GROUP_EMPTY)
    ## An empty group that contains no processes.

#[ error handling ]#

proc mpiCheck*(code: cint) =
  ## Asserts that an MPI function call was successful. If the code is not `MPI_SUCCESS`,
  ## raises an `MpiError` with the corresponding error message.
  ## 
  ## Parameters:
  ##  - `code`: The return code from an MPI function call to check for success.
  ## 
  ## Example:
  ## ```nim
  ## var rank: cint
  ## mpiCheck MPI_Comm_rank(WorldCommunicator.comm, addr rank)
  ## ```
  if code != MPI_SUCCESS:
    var buf: array[1024, char]
    var len: cint
    discard MPI_Error_string(code, cast[cstring](addr buf[0]), addr len)
    var message = newString(len)
    copyMem(addr message[0], addr buf[0], len)
    var err = newException(MpiError, message)
    err.code = code
    raise err

#[ initialization/finalization ]#

proc mpiInit* = 
  ## Initializes the MPI environment. Must be called before any other MPI functions.
  mpiCheck MPI_Init(nil, nil)

proc mpiFinalize* = 
  ## Finalizes the MPI environment. Should be called after all MPI functions.
  mpiCheck MPI_Finalize()

proc mpiInitialized*: bool = 
  ## Checks if the MPI environment has been initialized.
  var flag: cint
  mpiCheck MPI_Initialized(addr flag)
  return flag != 0

proc mpiFinalized*: bool = 
  ## Checks if the MPI environment has been finalized.
  var flag: cint
  mpiCheck MPI_Finalized(addr flag)
  return flag != 0

#[ MPI operation converter ]#

proc mpiOp*(op: MpiOperation): MPI_Op =
  ## Converts an `MpiOperation` to the underlying `MPI_Op` handle.
  result = MPI_NO_OP
  if op == ReduceSum:        result = MPI_SUM
  elif op == ReduceProduct:  result = MPI_PROD
  elif op == ReduceMaximum:  result = MPI_MAX
  elif op == ReduceMinimum:  result = MPI_MIN
  elif op == ReduceLogicalAnd:  result = MPI_LAND
  elif op == ReduceBitwiseAnd:  result = MPI_BAND
  elif op == ReduceLogicalOr:   result = MPI_LOR
  elif op == ReduceBitwiseOr:   result = MPI_BOR
  elif op == ReduceLogicalXor:  result = MPI_LXOR
  elif op == ReduceBitwiseXor:  result = MPI_BXOR
  elif op == ReduceMaxLoc:   result = MPI_MAXLOC
  elif op == ReduceMinLoc:   result = MPI_MINLOC
  elif op == ReduceReplace:  result = MPI_REPLACE

#[ MPI type converter ]#

proc mpiType*(T: typedesc): MPI_Datatype =
  ## Converts a Nim type to the corresponding MPI datatype. This is used for 
  ## specifying the datatype in MPI communication operations.
  ## 
  ## Parameters:
  ##  - `T`: The Nim type to convert to an MPI datatype.
  ##
  ## Returns:
  ##  - The corresponding `MPI_Datatype` for the given Nim type.\
  when T is int8:    MPI_INT8_T
  elif T is int16:   MPI_INT16_T
  elif T is int32:   MPI_INT32_T
  elif T is int64:   MPI_INT64_T
  elif T is uint8:   MPI_UINT8_T
  elif T is uint16:  MPI_UINT16_T
  elif T is uint32:  MPI_UINT32_T
  elif T is uint64:  MPI_UINT64_T
  elif T is float32: MPI_FLOAT
  elif T is float64: MPI_DOUBLE
  elif T is cint:    MPI_INT
  elif T is cchar:   MPI_CHAR
  elif T is char:    MPI_CHAR
  elif T is byte:    MPI_BYTE
  elif T is int:
    when sizeof(int) == 4: MPI_INT32_T
    else:                  MPI_INT64_T
  elif T is uint:
    when sizeof(uint) == 4: MPI_UINT32_T
    else:                   MPI_UINT64_T
  elif T is float:
    when sizeof(float) == 4: MPI_FLOAT
    else:                    MPI_DOUBLE
  else:
    {.error: "No MPI_Datatype mapping for " & $T.}

#[ timers ]#

proc mpiWallTime*: float64 = 
  ## Returns the current wall clock time in seconds. The time is measured from an 
  ## arbitrary point in the past, and is only meaningful when compared to another 
  ## value returned by `mpiWallTime`. This function is typically used for timing 
  ## code sections in MPI programs.
  return MPI_Wtime()

proc mpiWallTick*: float64 = 
  ## Returns the resolution of the timer used by `mpiWallTime`, in seconds. This value
  ## represents the smallest measurable time interval that can be returned by `mpiWallTime`.
  return MPI_Wtick()

#[ group ]#

proc group*(communicator: MpiCommunicator): MpiGroup =
  ## Returns the group associated with the given communicator.
  ## 
  ## Parameters:
  ##  - `communicator`: The communicator for which to retrieve the group.
  ## 
  ## Returns:
  ##  - An `MpiGroup` object representing the group associated with the communicator.
  var group: mpiwrap.MPI_Group
  mpiCheck MPI_Comm_group(communicator.comm, addr group)
  return MpiGroup(group: group)

proc newMpiGroup*(ranks: seq[int]): MpiGroup =
  ## Creates a new group from a sequence of ranks. The ranks should be specified with respect to `MPI_COMM_WORLD`.
  ##
  ## Parameters:
  ##  - `ranks`: A sequence of integer ranks that specify the processes to include in the new group. Ranks should be specified with respect to `MPI_COMM_WORLD`.
  ## 
  ## Returns:
  ##  - An `MpiGroup` object representing the new group that includes the specified ranks.
  var group: mpiwrap.MPI_Group
  var cRanks = newSeq[cint](ranks.len)
  for i in 0..<ranks.len: cRanks[i] = cint(ranks[i])
  mpiCheck MPI_Group_incl(
    WorldCommunicator.group.group, 
    cint(ranks.len), 
    addr cRanks[0], 
    addr group
  )
  return MpiGroup(group: group)

proc newMpiGroup*(group: MpiGroup; ranks: seq[int]): MpiGroup =
  ## Creates a new group from a sequence of ranks. The ranks should be specified with respect to the given group.
  ## 
  ## Parameters:
  ##  - `group`: The `MpiGroup` with respect to which the ranks are specified.
  ##  - `ranks`: A sequence of integer ranks that specify the processes to include in the new group. Ranks should be specified with respect to the given group.
  ##
  ## Returns:
  ##  - An `MpiGroup` object representing the new group that includes the specified ranks.
  var newGroup: mpiwrap.MPI_Group
  var cRanks = newSeq[cint](ranks.len)
  for i in 0..<ranks.len: cRanks[i] = cint(ranks[i])
  mpiCheck MPI_Group_incl(group.group, cint(ranks.len), addr cRanks[0], addr newGroup)
  return MpiGroup(group: newGroup)

proc size*(group: MpiGroup): int =
  ## Returns the size of the group (number of ranks in the group).
  ## 
  ## Parameters:
  ##  - `group`: The `MpiGroup` for which to retrieve the size.
  ## 
  ## Returns:
  ##  - The size of the group (number of ranks in the group).
  var size: cint
  mpiCheck MPI_Group_size(group.group, addr size)
  return int(size)

proc myRank*(group: MpiGroup): int =
  ## Returns the rank of the calling process within the group, or `MPI_UNDEFINED` if the process is not a member of the group.
  ## 
  ## Parameters:
  ##  - `group`: The `MpiGroup` for which to retrieve the rank.
  ## 
  ## Returns:
  ##  - The rank of the calling process within the group, or `MPI_UNDEFINED` if the process is not a member of the group.
  var rank: cint
  mpiCheck MPI_Group_rank(group.group, addr rank)
  return int(rank)

proc free*(group: var MpiGroup) =
  ## Frees the group. This routine does not free group storage, which is freed only when
  ## all references to the group are removed. 
  ## 
  ## Parameters:
  ## - `group`: The `MpiGroup` to free.
  mpiCheck MPI_Group_free(addr group.group)
  group.group = MPI_GROUP_NULL

proc `+`*(group1, group2: MpiGroup): MpiGroup =
  ## Returns the union of two groups, which contains all processes that are in either group.
  ## 
  ## Parameters:
  ##  - `group1`: The first group to union.
  ##  - `group2`: The second group to union.
  ## 
  ## Returns:
  ##  - An `MpiGroup` representing the union of the two groups.
  ## 
  ## Example:
  ## ```nim
  ## let group1 = newMpiGroup(@[0, 2])
  ## let group2 = newMpiGroup(@[0, 1, 3])
  ## let unionGroup = group1 + group2  # unionGroup includes ranks 0, 1, 2, 3
  ## ```
  var newGroup: mpiwrap.MPI_Group
  mpiCheck MPI_Group_union(group1.group, group2.group, addr newGroup)
  return MpiGroup(group: newGroup)

proc `*`*(group1, group2: MpiGroup): MpiGroup =
  ## Returns the intersection of two groups, which contains only processes that are in both groups.
  ## 
  ## Parameters:
  ##  - `group1`: The first group to intersect.
  ##  - `group2`: The second group to intersect.
  ## 
  ## Returns:
  ##  - An `MpiGroup` representing the intersection of the two groups.
  ## 
  ## Example:
  ## ```nim
  ## let group1 = newMpiGroup(@[0, 2])
  ## let group2 = newMpiGroup(@[0, 1, 3])
  ## let intersectionGroup = group1 * group2  # intersectionGroup includes only rank 0
  ## ```
  var newGroup: mpiwrap.MPI_Group
  mpiCheck MPI_Group_intersection(group1.group, group2.group, addr newGroup)
  return MpiGroup(group: newGroup)

proc `-`*(group1, group2: MpiGroup): MpiGroup =
  ## Returns the difference of two groups, which contains processes that are in `group1` but not in `group2`.
  ## 
  ## Parameters:
  ##  - `group1`: The group from which to subtract.
  ##  - `group2`: The group to subtract from `group1`.
  ## 
  ## Returns:
  ##  - An `MpiGroup` representing the difference of the two groups.
  ## 
  ## Example:
  ## ```nim
  ## let group1 = newMpiGroup(@[0, 2])
  ## let group2 = newMpiGroup(@[0, 1, 3])
  ## let differenceGroup = group1 - group2  # differenceGroup includes only rank 2
  ## ```
  var newGroup: mpiwrap.MPI_Group
  mpiCheck MPI_Group_difference(group1.group, group2.group, addr newGroup)
  return MpiGroup(group: newGroup)

#[ communicator ]#

proc `=destroy`(communicator: var MpiCommunicator) =
  ## Destructor hook for user-created communicators
  ## 
  ## Does not free communicators that are `MPI_COMM_NULL`, `MPI_COMM_WORLD`, or 
  ## `MPI_COMM_SELF`, as these are managed by MPI and should not be freed by the user. 
  ## For other communicators, checks if MPI is initialized and not finalized before 
  ## freeing the communicator. This ensures that communicators are only freed when 
  ## it is safe to do so.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to be destroyed.
  if (
    communicator.comm != MPI_COMM_NULL and
    communicator.comm != MPI_COMM_WORLD and
    communicator.comm != MPI_COMM_SELF
  ):
    var isInit: cint
    var isFinalized: cint
    discard MPI_Initialized(addr isInit)
    discard MPI_Finalized(addr isFinalized)
    if isInit != 0 and isFinalized == 0:
      discard MPI_Comm_free(addr communicator.comm)
  communicator.comm = MPI_COMM_NULL

proc newMpiCommunicator*(
  communicator: MpiCommunicator; 
  group: MpiGroup
): MpiCommunicator =
  ## Creates a new communicator from a group. The new communicator will include only the processes in the given group.
  var comm: MPI_Comm
  mpiCheck MPI_Comm_create(communicator.comm, group.group, addr comm)
  return MpiCommunicator(comm: comm)

proc newMpiCommunicator*(group: MpiGroup): MpiCommunicator =
  ## Creates a new communicator from a group. The new communicator will include only the processes in the given group.
  var comm: MPI_Comm
  mpiCheck MPI_Comm_create(WorldCommunicator.comm, group.group, addr comm)
  return MpiCommunicator(comm: comm)

proc duplicate*(communicator: MpiCommunicator): MpiCommunicator =
  ## Duplicates the given communicator
  ## 
  ## Parameters:
  ##  - `communicator`: The communicator to duplicate.
  ## 
  ## Returns:
  ##  - A new `MpiCommunicator` representing the duplicated communicator.
  ## 
  ## Example:
  ## ```nim
  ## let comm = WorldCommunicator
  ## let dupComm = comm.duplicate()
  ## echo "Original communicator rank: ", comm.myRank, ", Duplicated communicator rank: ", dupComm.myRank
  ## ```
  var comm: MPI_Comm
  mpiCheck MPI_Comm_dup(communicator.comm, addr comm)
  return MpiCommunicator(comm: comm)

proc split*(communicator: MpiCommunicator; color, key: int): MpiCommunicator =
  ## Creates a subcommunicator from `communicator` using `MPI_Comm_split`.
  ##
  ## Splits the given communicator into one or more subcommunicators based on 
  ## the "color" and "key" values. "Color" determines the subcommunicator assignment; 
  ## i.e., all processes passing the same value for "color" will be in the same 
  ## subcommunicator. "Key" determines the rank ordering within the new communicator; 
  ## i.e., processes with lower key values will have lower ranks in the new 
  ## communicator - the processes with the lowest key is therefore assigned to rank 0.
  ## 
  ## Parameters:
  ##  - `communicator`: The communicator to split.
  ##  - `color`: Used to determine subcommunicator assignment. 
  ##  - `key`: Used to determine rank ordering within the new communicator.
  ## 
  ## Returns:
  ##  - A new `MpiCommunicator` subcommunicator
  ## 
  ## Example:
  ## ```nim
  ## let comm = WorldCommunicator
  ## let subcomm = comm.split(comm.myRank mod 2, comm.myRank)
  ## echo "Process ", comm.myRank, " is in subcommunicator with rank ", subcomm.myRank
  ## ```
  var comm: MPI_Comm
  mpiCheck MPI_Comm_split(communicator.comm, cint(color), cint(key), addr comm)
  return MpiCommunicator(comm: comm)

proc abort*(communicator: MpiCommunicator, errorcode: int = 1) =
  ## Aborts the MPI program with the given error code.
  ## 
  ## Parameters:
  ##  - `communicator`: The communicator to use for the abort operation.
  ##  - `errorcode`: The error code to return upon aborting (default is 1).
  ## 
  ## Example:
  ## ```nim
  ## let comm = WorldCommunicator
  ## if comm.myRank == 0:
  ##   comm.abort(42)  # Aborts the program with error code 42
  ## ```
  mpiCheck MPI_Abort(communicator.comm, cint(errorcode))

proc free*(communicator: var MpiCommunicator) =
  ## Frees communicator
  ## 
  ## This routine does not free communicator storage, which is freed only when
  ## all references to the communictor are removed. 
  ## 
  ## Parameters:
  ##  - `communicator`: The communicator to free. 
  mpiCheck MPI_Comm_free(addr communicator.comm)

proc size*(communicator: MpiCommunicator = WorldCommunicator): int =
  ## Returns communicator size (number of ranks in the communicator)
  ## 
  ## Parameters:
  ##  - `communicator`: MpiCommunicator object
  ##
  ## Returns:
  ##  - The size of the communicator (number of ranks). 
  var size: cint
  mpiCheck MPI_Comm_size(communicator.comm, addr size)
  return int(size)

proc myRank*(communicator: MpiCommunicator = WorldCommunicator): int =
  ## Returns the rank of the calling process within the given communicator.
  ## 
  ## Note that ranks are zero-indexed.
  ## 
  ## Parameters:
  ##  - `communicator`: MpiCommunicator object
  ## 
  ## Returns:
  ##  - The rank of the calling process within the communicator.
  var rank: cint
  mpiCheck MPI_Comm_rank(communicator.comm, addr rank)
  return int(rank)

proc mpiBarrier*(communicator: MpiCommunicator = WorldCommunicator) =
  ## Blocks until all processes in the communicator have reached this routine. 
  ## This is a collective operation that synchronizes all processes in the communicator.
  ##
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` for which to perform the barrier operation.
  ##
  ## Example:
  ## ```nim
  ## let comm = WorldCommunicator
  ## echo "Process ", comm.myRank, " before barrier"
  ## comm.mpiBarrier()
  ## echo "Process ", comm.myRank, " after barrier"
  ## ```
  mpiCheck MPI_Barrier(communicator.comm)

proc `==`*(comm1, comm2: MpiCommunicator): bool =
  ## Compares two communicators for equality. Two communicators are considered equal if they are handles for the same underlying MPI communicator.
  ## 
  ## Parameters:
  ##  - `comm1`: The first `MpiCommunicator` to compare.
  ##  - `comm2`: The second `MpiCommunicator` to compare.
  ##
  ## Returns:
  ##  - `true` if the communicators are equal (i.e., they refer to the same underlying MPI communicator), and `false` otherwise.
  var flag: cint
  mpiCheck MPI_Comm_compare(comm1.comm, comm2.comm, addr flag)
  return flag == MPI_IDENT

proc probe*(communicator: MpiCommunicator; source: int; tag: int = 0): MpiStatus =
  ## Probes for an incoming message from the process with rank `source` in `communicator` with the given `tag`.
  ## This is a blocking operation that waits until a matching message is available. The returned `MpiStatus` contains
  ## information about the source, tag, and error code of the probed message.
  ##
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to probe within.
  ##  - `source`: The rank of the source process to probe for (or `MPI_ANY_SOURCE` to probe for messages from any source).
  ##  - `tag`: The tag of the message to probe for (or `MPI_ANY_TAG` to probe for messages with any tag).
  ##
  ## Returns:
  ##  - An `MpiStatus` object containing information about the probed message, including the source, tag, and error code.
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Probe(cint(source), cint(tag), communicator.comm, addr rawStatus)
  return MpiStatus(status: rawStatus)

proc immediateProbe*(communicator: MpiCommunicator; source: int; tag: int = 0): (bool, MpiStatus) =
  ## Probes for an incoming message from the process with rank `source` in `communicator` with the given `tag` without blocking.
  ## This is a non-blocking operation that checks if a matching message is available. If a matching message is found, returns `true` and an `MpiStatus` containing information about the message. If no matching message is available, returns `false` and an undefined `MpiStatus`.
  ##
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to probe within.
  ##  - `source`: The rank of the source process to probe for (or `MPI_ANY_SOURCE` to probe for messages from any source).
  ##  - `tag`: The tag of the message to probe for (or `MPI_ANY_TAG` to probe for messages with any tag).
  ##
  ## Returns:
  ##  - A tuple `(found, status)`. When `found` is `true`, a matching message was found and `status` contains information about the message. When `found` is `false`, no matching message is available and `status` is undefined.
  var flag: cint
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Iprobe(cint(source), cint(tag), communicator.comm, addr flag, addr rawStatus)
  return (flag != 0, MpiStatus(status: rawStatus))

macro echo*(communicator: MpiCommunicator; message: varargs[untyped]): untyped =
  ## Prints `message` from rank 0 of `communicator` only.
  ## Accepts the same comma-separated arguments as the built-in `echo`.
  ##
  ## Example:
  ## ```nim
  ## WorldCommunicator.echo "rank 0 says hi"
  ## customComm.echo "size = ", customComm.size
  ## ```
  result = quote do:
    if `communicator`.myRank() == 0: echo `message`

#[ status ]#

proc source*(status: MpiStatus): int =
  ## Returns the source rank from the given `MpiStatus`.
  ## 
  ## Parameters:
  ##  - `status`: The `MpiStatus` from which to retrieve the source rank.
  ##
  ## Returns:
  ##  - The source rank associated with the status.
  return int(status.status.MPI_SOURCE)

proc tag*(status: MpiStatus): int =
  ## Returns the tag from the given `MpiStatus`.
  ## 
  ## Parameters:
  ##  - `status`: The `MpiStatus` from which to retrieve the tag.
  ## 
  ## Returns:
  ##  - The tag associated with the status.
  return int(status.status.MPI_TAG)

proc error*(status: MpiStatus): int =
  ## Returns the error code from the given `MpiStatus`.
  ## 
  ## Parameters:
  ##  - `status`: The `MpiStatus` from which to retrieve the error code.
  ##
  ## Returns:
  ##  - The error code associated with the status.
  return int(status.status.MPI_ERROR)

proc count*[T](status: MpiStatus): int =
  ## Returns the number of received elements of type `T` from the given `MpiStatus`.
  ##
  ## Parameters:
  ##  - `status`: The `MpiStatus` from which to retrieve the element count.
  ##
  ## Returns:
  ##  - The number of elements of type `T` received.
  ##
  ## Example:
  ## ```nim
  ## let n = status.count[:float32]
  ## ```
  var c: cint
  mpiCheck MPI_Get_count(unsafeAddr status.status, mpiType(T), addr c)
  return int(c)

#[ request ]#

proc wait*(request: var MpiRequest): MpiStatus =
  ## Waits for the non-blocking operation associated with `request` to complete.
  ## 
  ## Parameters:
  ##  - `request`: The `MpiRequest` representing the non-blocking operation to wait for.
  ## 
  ## Returns:
  ##  - An `MpiStatus` containing the source, tag, and error code of the completed operation.
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Wait(addr request.request, addr rawStatus)
  request.request = MPI_REQUEST_NULL
  return MpiStatus(status: rawStatus)

proc test*(request: var MpiRequest): (bool, MpiStatus) =
  ## Tests if the non-blocking operation associated with `request` has completed.
  ## 
  ## Parameters:
  ##  - `request`: The `MpiRequest` representing the non-blocking operation to test.
  ##
  ## Returns:
  ##  - A tuple `(done, status)`. When `done` is `true`, the operation has completed and
  ##    `status` contains the source, tag, and error code. When `done` is `false`, `status`
  ##    is undefined and should not be used.
  var flag: cint
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Test(addr request.request, addr flag, addr rawStatus)
  if flag != 0: request.request = MPI_REQUEST_NULL
  return (flag != 0, MpiStatus(status: rawStatus))

proc wait*(requests: var seq[MpiRequest]): seq[MpiStatus] =
  ## Waits for all non-blocking operations in `requests` to complete.
  ## 
  ## Parameters:
  ##  - `requests`: A sequence of `MpiRequest` objects representing the non-blocking operations to wait for.
  ##
  ## Returns:
  ##  - A sequence of `MpiStatus` objects containing the source, tag, and error code for each completed operation. The order of statuses corresponds to the order of requests.
  var rawStatuses = newSeq[mpiwrap.MPI_Status](requests.len)
  var cRequests = newSeq[mpiwrap.MPI_Request](requests.len)
  for i in 0..<requests.len: cRequests[i] = requests[i].request
  mpiCheck MPI_Waitall(cint(requests.len), addr cRequests[0], addr rawStatuses[0])
  for i in 0..<requests.len: requests[i].request = MPI_REQUEST_NULL
  var statuses = newSeq[MpiStatus](requests.len)
  for i in 0..<requests.len: statuses[i] = MpiStatus(status: rawStatuses[i])
  return statuses

proc test*(requests: var seq[MpiRequest]): (bool, seq[MpiStatus]) =
  ## Tests if all non-blocking operations in `requests` have completed.
  ## 
  ## Parameters:
  ##  - `requests`: A sequence of `MpiRequest` objects representing the non-blocking operations to test.
  ##
  ## Returns:
  ##  - A tuple `(done, statuses)`. When `done` is `true`, all operations have completed and `statuses` contains the source, tag, and error code for each operation. When `done` is `false`, `statuses` is undefined and should not be used.
  var flag: cint
  var rawStatuses = newSeq[mpiwrap.MPI_Status](requests.len)
  var cRequests = newSeq[mpiwrap.MPI_Request](requests.len)
  for i in 0..<requests.len: cRequests[i] = requests[i].request
  mpiCheck MPI_Testall(cint(requests.len), addr cRequests[0], addr flag, addr rawStatuses[0])
  if flag != 0:
    for i in 0..<requests.len: requests[i].request = MPI_REQUEST_NULL
    var statuses = newSeq[MpiStatus](requests.len)
    for i in 0..<requests.len: statuses[i] = MpiStatus(status: rawStatuses[i])
    return (true, statuses)
  else:
    return (false, @[])

#[ point-to-point communication ]#

proc send*[T](
  communicator: MpiCommunicator;
  buffer: pointer;
  count: int;
  datatype: typedesc[T];
  dest: int;
  tag: int = 0;
  sendType: SendType = StandardSend
) =
  ## Sends data from `buffer` to the process with rank `dest` in `communicator` using the specified send type.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for sending.
  ##  - `buffer`: A pointer to the data to send.
  ##  - `count`: The number of elements to send.
  ##  - `datatype`: The MPI datatype of the elements to send.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ##  - `sendType`: The type of send operation to perform (default is `StandardSend`).
  case sendType
  of StandardSend:
    mpiCheck MPI_Send(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm
    )
  of BufferedSend:
    mpiCheck MPI_Bsend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm
    )
  of SynchronousSend:
    mpiCheck MPI_Ssend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm
    )
  of ReadySend:
    mpiCheck MPI_Rsend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm
    )

proc send*[T](
  communicator: MpiCommunicator;
  buffer: openArray[T];
  dest: int;
  tag: int = 0;
  sendType: SendType = StandardSend
) =
  ## Sends `buffer` to the process with rank `dest` in `communicator` using `MPI_Send`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for sending.
  ##  - `buffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  communicator.send(addr buffer[0], buffer.len, T, dest, tag, sendType)

proc immediateSend*[T](
  communicator: MpiCommunicator;
  buffer: pointer;
  count: int;
  datatype: typedesc[T];
  dest: int;
  tag: int = 0;
  sendType: SendType = StandardSend
): MpiRequest =
  ## Initiates a non-blocking send of data from `buffer` to the process with rank `dest` in `communicator` using the specified send type.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for sending.
  ##  - `buffer`: A pointer to the data to send.
  ##  - `count`: The number of elements to send.
  ##  - `datatype`: The MPI datatype of the elements to send.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ##  - `sendType`: The type of send operation to perform (default is `StandardSend`).
  var request: mpiwrap.MPI_Request
  case sendType
  of StandardSend:
    mpiCheck MPI_Isend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm, 
      addr request
    )
  of BufferedSend:
    mpiCheck MPI_Ibsend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm, 
      addr request
    )
  of SynchronousSend:
    mpiCheck MPI_Issend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm, 
      addr request
    )
  of ReadySend:
    mpiCheck MPI_Irsend(
      buffer, 
      cint(count), 
      mpiType(T), 
      cint(dest), 
      cint(tag), 
      communicator.comm, 
      addr request
    )
  return MpiRequest(request: request)

proc immediateSend*[T](
  communicator: MpiCommunicator;
  buffer: openArray[T];
  dest: int;
  tag: int = 0;
  sendType: SendType = StandardSend
): MpiRequest =
  ## Initiates a non-blocking send of `buffer` to the process with rank `dest` in `communicator` using `MPI_Isend`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for sending.
  ##  - `buffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ## 
  ## Returns:
  ##  - An `MpiRequest` object representing the non-blocking send operation, which can be used to test for completion or wait for completion.
  return communicator.immediateSend(
    addr buffer[0], 
    buffer.len, 
    T, 
    dest, 
    tag, 
    sendType
  )

proc receive*[T](
  communicator: MpiCommunicator; 
  buffer: pointer; 
  count: int; 
  datatype: typedesc[T]; 
  source: int; 
  tag: int = 0
): MpiStatus {.discardable.} =
  ## Receives data into `buffer` from the process with rank `source` in `communicator` using `MPI_Recv`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for receiving.
  ##  - `buffer`: A pointer to the buffer to receive data into.
  ##  - `count`: The number of elements to receive.
  ##  - `datatype`: The MPI datatype of the elements to receive.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ## 
  ## Returns:
  ##  - An `MpiStatus` containing the source, tag, and error code of the received message.
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Recv(
    buffer, 
    cint(count), 
    mpiType(T), 
    cint(source), 
    cint(tag), 
    communicator.comm, 
    addr rawStatus
  )
  return MpiStatus(status: rawStatus)

proc receive*[T](
  communicator: MpiCommunicator; 
  buffer: var openArray[T]; 
  source: int; 
  tag: int = 0
): MpiStatus {.discardable.} =
  ## Receives data into `buffer` from the process with rank `source` in `communicator` using `MPI_Recv`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for receiving.
  ##  - `buffer`: The buffer to receive data into. Can be any type that has a corresponding MPI datatype.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ## 
  ## Returns:
  ##  - An `MpiStatus` containing the source, tag, and error code of the received message.
  return communicator.receive(addr buffer[0], buffer.len, T, source, tag)

proc immediateReceive*[T](
  communicator: MpiCommunicator; 
  buffer: pointer; 
  count: int; 
  datatype: typedesc[T]; 
  source: int; 
  tag: int = 0
): MpiRequest =
  ## Initiates a non-blocking receive into `buffer` from the process with rank `source` in `communicator` using `MPI_Irecv`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for receiving.
  ##  - `buffer`: A pointer to the buffer to receive data into.
  ##  - `count`: The number of elements to receive.
  ##  - `datatype`: The MPI datatype of the elements to receive.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ## 
  ## Returns:
  ##  - An `MpiRequest` object representing the non-blocking receive operation, which can be used to test for completion or wait for completion.
  var request: mpiwrap.MPI_Request
  mpiCheck MPI_Irecv(
    buffer, 
    cint(count), 
    mpiType(T), 
    cint(source), 
    cint(tag), 
    communicator.comm, 
    addr request
  )
  return MpiRequest(request: request)

proc immediateReceive*[T](
  communicator: MpiCommunicator; 
  buffer: var openArray[T]; 
  source: int; 
  tag: int = 0
): MpiRequest =
  ## Initiates a non-blocking receive into `buffer` from the process with rank `source` in `communicator` using `MPI_Irecv`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for receiving.
  ##  - `buffer`: The buffer to receive data into. Can be any type that has a corresponding MPI datatype.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `tag`: An optional tag to identify the message (default is 0).
  ## 
  ## Returns:
  ##  - An `MpiRequest` object representing the non-blocking receive operation, which can be used to test for completion or wait for completion.
  return communicator.immediateReceive(addr buffer[0], buffer.len, T, source, tag)

proc sendReceive*[T](
  communicator: MpiCommunicator;
  sendBuffer, recvBuffer: pointer;
  sendCount, recvCount: int;
  sendDatatype, recvDatatype: typedesc[T];
  dest, source: int;
  sendTag: int = 0;
  recvTag: int = 0
): MpiStatus {.discardable.} =
  ## Performs a combined send and receive operation, sending data from `sendBuffer` to the process with rank `dest` and receiving data into `recvBuffer` from the process with rank `source` in `communicator`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `sendBuffer`: A pointer to the data to send.
  ##  - `recvBuffer`: A pointer to the buffer to receive data into.
  ##  - `sendCount`: The number of elements to send.
  ##  - `recvCount`: The number of elements to receive.
  ##  - `sendDatatype`: The MPI datatype of the elements to send.
  ##  - `recvDatatype`: The MPI datatype of the elements to receive.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `sendTag`: The tag to identify the sent message.
  ##  - `recvTag`: The tag to identify the received message.
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Sendrecv(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    cint(dest), 
    cint(sendTag), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    cint(source), 
    cint(recvTag), 
    communicator.comm, 
    addr rawStatus
  )
  return MpiStatus(status: rawStatus)

proc sendReceive*[T](
  communicator: MpiCommunicator;
  sendBuffer, recvBuffer: openArray[T];
  dest, source: int;
  sendTag: int = 0;
  recvTag: int = 0
): MpiStatus {.discardable.} =
  ## Performs a combined send and receive operation, sending `sendBuffer` to the process with rank `dest` and receiving data into `recvBuffer` from the process with rank `source` in `communicator`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive data into. Can be any type that has a corresponding MPI datatype.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `sendTag`: The tag to identify the sent message.
  ##  - `recvTag`: The tag to identify the received message.
  return communicator.sendReceive(
    addr sendBuffer[0], 
    addr recvBuffer[0], 
    sendBuffer.len, 
    recvBuffer.len, 
    T, 
    T, 
    dest, 
    source, 
    sendTag, 
    recvTag
  )

proc sendReceiveReplace*[T](
  communicator: MpiCommunicator;
  buffer: pointer;
  count: int;
  datatype: typedesc[T];
  dest, source: int;
  sendTag: int = 0;
  recvTag: int = 0
): MpiStatus {.discardable.} =
  ## Performs a combined send and receive operation, sending data from `buffer` to the process with rank `dest` and receiving data into the same `buffer` from the process with rank `source` in `communicator`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `buffer`: A pointer to the data to send and receive.
  ##  - `count`: The number of elements to send and receive.
  ##  - `datatype`: The MPI datatype of the elements to send and receive.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `sendTag`: The tag to identify the sent message.
  ##  - `recvTag`: The tag to identify the received message.
  var rawStatus: mpiwrap.MPI_Status
  mpiCheck MPI_Sendrecv_replace(
    buffer, 
    cint(count), 
    mpiType(T), 
    cint(dest), 
    cint(sendTag), 
    cint(source), 
    cint(recvTag), 
    communicator.comm, 
    addr rawStatus
  )
  return MpiStatus(status: rawStatus)

proc sendReceiveReplace*[T](
  communicator: MpiCommunicator;
  buffer: var openArray[T];
  dest, source: int;
  sendTag: int = 0;
  recvTag: int = 0
): MpiStatus {.discardable.} =
  ## Performs a combined send and receive operation, sending `buffer` to the process with rank `dest` and receiving data into the same `buffer` from the process with rank `source` in `communicator`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `buffer`: The data to send and receive. Can be any type that has a corresponding MPI datatype.
  ##  - `dest`: The rank of the destination process within the communicator.
  ##  - `source`: The rank of the source process within the communicator.
  ##  - `sendTag`: The tag to identify the sent message.
  ##  - `recvTag`: The tag to identify the received message.
  ## 
  ## Returns:
  ##  - An `MpiStatus` containing the source, tag, and error code of the received message.
  return communicator.sendReceiveReplace(
    addr buffer[0], 
    buffer.len, 
    T, 
    dest, 
    source, 
    sendTag, 
    recvTag
  )

#[ collective operations ]#

proc broadcast*[T](
  communicator: MpiCommunicator; 
  buffer: pointer; 
  count: int; 
  datatype: typedesc[T]; 
  root: int
) =
  ## Broadcasts data from the process with rank `root` to all other processes in `communicator` using `MPI_Bcast`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for broadcasting.
  ##  - `buffer`: A pointer to the data to broadcast. Can be any type that has a corresponding MPI datatype.
  ##  - `count`: The number of elements to broadcast.
  ##  - `datatype`: The MPI datatype of the elements to broadcast.
  ##  - `root`: The rank of the root process that will broadcast the data.
  mpiCheck MPI_Bcast(
    buffer, 
    cint(count), 
    mpiType(T), 
    cint(root), 
    communicator.comm
  )

proc broadcast*[T](
  communicator: MpiCommunicator; 
  buffer: var openArray[T]; 
  root: int
) =
  ## Broadcasts `buffer` from the process with rank `root` to all other processes in `communicator` using `MPI_Bcast`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for broadcasting.
  ##  - `buffer`: The data to broadcast. Can be any type that has a corresponding MPI datatype.
  ##  - `root`: The rank of the root process that will broadcast the data.
  communicator.broadcast(addr buffer[0], buffer.len, T, root)

proc scatter*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCount, recvCount: int; 
  sendDatatype, recvDatatype: typedesc[T];
  root: int
) =
  ## Scatters data from the process with rank `root` to all other processes in `communicator` using `MPI_Scatter`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for scattering.
  ##  - `sendBuffer`: A pointer to the data to scatter. Only significant at the root process.
  ##  - `recvBuffer`: A pointer to the buffer to receive the scattered data into.
  ##  - `sendCount`: The number of elements to send to each process (significant at root).
  ##  - `recvCount`: The number of elements to receive from the root.
  ##  - `sendDatatype`: The MPI datatype of the elements to send (significant at root).
  ##  - `recvDatatype`: The MPI datatype of the elements to receive.
  ##  - `root`: The rank of the root process that will scatter the data.
  mpiCheck MPI_Scatter(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    cint(root), 
    communicator.comm
  )

proc scatter*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T]; 
  root: int;
  sendLen, recvLen: int
) =
  ## Scatters data from the process with rank `root` to all other processes in `communicator` using `MPI_Scatter`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for scattering.
  ##  - `sendBuffer`: The data to scatter. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the scattered data into. Can be any type that has a corresponding MPI datatype.
  ##  - `root`: The rank of the root process that will scatter the data.
  ##  - `sendLen`: The number of elements to send to each process (significant at root).
  ##  - `recvLen`: The number of elements to receive from the root.
  communicator.scatter(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendLen,
    recvLen,
    T, T,
    root
  )

proc variableScatter*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCounts: openArray[int];
  recvCount: int;
  sendDatatype, recvDatatype: typedesc[T]; 
  displacements: openArray[int];
  root: int
) =
  ## Scatters data from the process with rank `root` to all other processes in `communicator` using `MPI_Scatterv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for scattering.
  ##  - `sendBuffer`: A pointer to the data to scatter. Only significant at the root process.
  ##  - `recvBuffer`: A pointer to the buffer to receive the scattered data into.
  ##  - `sendCounts`: An array of counts specifying the number of elements to send to each process (significant at root).
  ##  - `recvCount`: The number of elements to receive from the root.
  ##  - `sendDatatype`: The MPI datatype of the elements to send (significant at root).
  ##  - `recvDatatype`: The MPI datatype of the elements to receive.
  ##  - `displacements`: An array of displacements specifying the starting index in `sendBuffer` for each process (significant at root).
  ##  - `root`: The rank of the root process that will scatter the data.
  var sc = newSeq[cint](sendCounts.len)
  for i in 0..<sendCounts.len: sc[i] = cint(sendCounts[i])
  var d = newSeq[cint](displacements.len)
  for i in 0..<displacements.len: d[i] = cint(displacements[i])
  mpiCheck MPI_Scatterv(
    sendBuffer, 
    if sc.len > 0: addr sc[0] else: nil,
    if d.len > 0: addr d[0] else: nil,
    mpiType(T), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    cint(root), 
    communicator.comm
  )

proc variableScatter*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  sendCounts, displacements: openArray[int]; 
  recvBuffer: var openArray[T]; 
  root: int;
  recvLen: int
) =
  ## Scatters data from the process with rank `root` to all other processes in `communicator` using `MPI_Scatterv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for scattering.
  ##  - `sendBuffer`: The data to scatter. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `sendCounts`: An array of counts specifying the number of elements to send to each process (significant at root).
  ##  - `displacements`: An array of displacements specifying the starting index in `sendBuffer` for each process (significant at root).
  ##  - `recvBuffer`: The buffer to receive the scattered data into. Can be any type that has a corresponding MPI datatype.
  ##  - `root`: The rank of the root process that will scatter the data.
  ##  - `recvLen`: The number of elements to receive from the root.
  communicator.variableScatter(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendCounts,
    recvLen,
    T, T,
    displacements,
    root
  )

proc gather*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCount, recvCount: int; 
  sendDatatype, recvDatatype: typedesc[T]; 
  root: int
) =
  ## Gathers data from all processes in `communicator` to the process with rank `root` using `MPI_Gather`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: A pointer to the data to send.
  ##  - `recvBuffer`: A pointer to the buffer to receive the gathered data into. Only significant at the root process.
  ##  - `sendCount`: The number of elements to send from each process.
  ##  - `recvCount`: The number of elements received from each process (significant at root).
  ##  - `sendDatatype`: The MPI datatype of the elements to send.
  ##  - `recvDatatype`: The MPI datatype of the elements received (significant at root).
  ##  - `root`: The rank of the root process that will gather the data.
  mpiCheck MPI_Gather(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    cint(root), 
    communicator.comm
  )

proc gather*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T]; 
  root: int;
  sendLen, recvLen: int
) =
  ## Gathers data from all processes in `communicator` to the process with rank `root` using `MPI_Gather`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the gathered data into. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `root`: The rank of the root process that will gather the data.
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process (significant at root).
  communicator.gather(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendLen,
    recvLen,
    T, T,
    root
  )

proc allGather*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCount, recvCount: int; 
  sendDatatype, recvDatatype: typedesc[T]; 
) =
  ## Gathers data from all processes in `communicator` and distributes the combined data to all processes using `MPI_Allgather`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: A pointer to the data to send.
  ##  - `recvBuffer`: A pointer to the buffer to receive the gathered data into.
  ##  - `sendCount`: The number of elements to send from each process.
  ##  - `recvCount`: The number of elements received from each process.
  ##  - `sendDatatype`: The MPI datatype of the elements to send.
  ##  - `recvDatatype`: The MPI datatype of the elements received.
  mpiCheck MPI_Allgather(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    communicator.comm
  )

proc allGather*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T]; 
  sendLen, recvLen: int
) =
  ## Gathers data from all processes in `communicator` and distributes the combined data to all processes using `MPI_Allgather`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the gathered data into. Can be any type that has a corresponding MPI datatype.
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process.
  communicator.allGather(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendLen,
    recvLen,
    T, T
  )

proc variableGather*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCount: int; 
  recvCounts: openArray[int];
  sendDatatype, recvDatatype: typedesc[T]; 
  displacements: openArray[int]; 
  root: int
) =
  ## Gathers data from all processes in `communicator` to the process with rank `root` using `MPI_Gatherv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: A pointer to the data to send.
  ##  - `recvBuffer`: A pointer to the buffer to receive the gathered data into. Only significant at the root process.
  ##  - `sendCount`: The number of elements to send from each process.
  ##  - `recvCounts`: An array of counts specifying the number of elements received from each process (significant at root).
  ##  - `sendDatatype`: The MPI datatype of the elements to send.
  ##  - `recvDatatype`: The MPI datatype of the elements received (significant at root).
  ##  - `displacements`: An array of displacements specifying the starting index in `recvBuffer` for each process (significant at root).
  ##  - `root`: The rank of the root process that will gather the data.
  var rc = newSeq[cint](recvCounts.len)
  for i in 0..<recvCounts.len: rc[i] = cint(recvCounts[i])
  var d = newSeq[cint](displacements.len)
  for i in 0..<displacements.len: d[i] = cint(displacements[i])
  mpiCheck MPI_Gatherv(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    recvBuffer, 
    if rc.len > 0: addr rc[0] else: nil,
    if d.len > 0: addr d[0] else: nil,
    mpiType(T), 
    cint(root), 
    communicator.comm
  )

proc variableGather*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T]; 
  recvCounts, displacements: openArray[int]; 
  root: int
) =
  ## Gathers data from all processes in `communicator` to the process with rank `root` using `MPI_Gatherv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for gathering.
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the gathered data into. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `recvCounts`: An array of counts specifying the number of elements received from each process (significant at root).
  ##  - `displacements`: An array of displacements specifying the starting index in `recvBuffer` for each process (significant at root).
  ##  - `root`: The rank of the root process that will gather the data.
  communicator.variableGather(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendBuffer.len,
    recvCounts,
    T, T,
    displacements,
    root
  )

proc allToAll*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCount, recvCount: int; 
  sendDatatype, recvDatatype: typedesc[T]
) =
  ## Performs an all-to-all communication, where each process sends data from `sendBuffer` to all other processes and receives data into `recvBuffer` from all other processes in `communicator` using `MPI_Alltoall`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `sendBuffer`: A pointer to the data to send.
  ##  - `recvBuffer`: A pointer to the buffer to receive data into.
  ##  - `sendCount`: The number of elements to send to each process.
  ##  - `recvCount`: The number of elements received from each process.
  ##  - `sendDatatype`: The MPI datatype of the elements to send.
  ##  - `recvDatatype`: The MPI datatype of the elements received.
  mpiCheck MPI_Alltoall(
    sendBuffer, 
    cint(sendCount), 
    mpiType(T), 
    recvBuffer, 
    cint(recvCount), 
    mpiType(T), 
    communicator.comm
  )

proc allToAll*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T]; 
  sendLen, recvLen: int
) =
  ## Performs an all-to-all communication, where each process sends `sendBuffer` to all other processes and receives data into `recvBuffer` from all other processes in `communicator` using `MPI_Alltoall`.
  ## 
  ## Parameters:
  ## - `communicator`: The `MpiCommunicator` to use for the operation.
  ## - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ## - `recvBuffer`: The buffer to receive data into. Can be any type that has a corresponding MPI datatype.
  ## - `sendLen`: The number of elements to send to each process.
  ## - `recvLen`: The number of elements received from each process.
  communicator.allToAll(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendLen,
    recvLen,
    T, T
  )

proc variableAllToAll*[T](
  communicator: MpiCommunicator; 
  sendBuffer, recvBuffer: pointer; 
  sendCounts, recvCounts: openArray[int]; 
  sendDisplacements, recvDisplacements: openArray[int]; 
  sendDatatype, recvDatatype: typedesc[T]
) =
  ## Performs a variable all-to-all communication, where each process sends data from `sendBuffer` to all other processes and receives data into `recvBuffer` from all other processes in `communicator` using `MPI_Alltoallv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ## - `communicator`: The `MpiCommunicator` to use for the operation.
  ## - `sendBuffer`: A pointer to the data to send.
  ## - `recvBuffer`: A pointer to the buffer to receive data into.
  ## - `sendCounts`: An array of counts specifying the number of elements to send to each process.
  ## - `recvCounts`: An array of counts specifying the number of elements received from each process.
  ## - `sendDisplacements`: An array of displacements specifying the starting index in `sendBuffer` for each process.
  ## - `recvDisplacements`: An array of displacements specifying the starting index in `recvBuffer` for each process.
  ## - `sendDatatype`: The MPI datatype of the elements to send.
  ## - `recvDatatype`: The MPI datatype of the elements received.
  var sc = newSeq[cint](sendCounts.len)
  for i in 0..<sendCounts.len: sc[i] = cint(sendCounts[i])
  var rc = newSeq[cint](recvCounts.len)
  for i in 0..<recvCounts.len: rc[i] = cint(recvCounts[i])
  var sd = newSeq[cint](sendDisplacements.len)
  for i in 0..<sendDisplacements.len: sd[i] = cint(sendDisplacements[i])
  var rd = newSeq[cint](recvDisplacements.len)
  for i in 0..<recvDisplacements.len: rd[i] = cint(recvDisplacements[i])
  mpiCheck MPI_Alltoallv(
    sendBuffer, 
    addr sc[0],
    addr sd[0],
    mpiType(T), 
    recvBuffer, 
    addr rc[0],
    addr rd[0],
    mpiType(T), 
    communicator.comm
  )

proc variableAllToAll*[T](
  communicator: MpiCommunicator; 
  sendBuffer: openArray[T]; 
  sendCounts, sendDisplacements: openArray[int]; 
  recvBuffer: var openArray[T]; 
  recvCounts, recvDisplacements: openArray[int]
) =
  ## Performs a variable all-to-all communication, where each process sends `sendBuffer` to all other processes and receives data into `recvBuffer` from all other processes in `communicator` using `MPI_Alltoallv`, allowing for variable counts and displacements.
  ## 
  ## Parameters:
  ## - `communicator`: The `MpiCommunicator` to use for the operation.
  ## - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ## - `sendCounts`: An array of counts specifying the number of elements to send to each process.
  ## - `sendDisplacements`: An array of displacements specifying the starting index in `sendBuffer` for each process.
  ## - `recvBuffer`: The buffer to receive data into. Can be any type that has a corresponding MPI datatype.
  ## - `recvCounts`: An array of counts specifying the number of elements received from each process.
  ## - `recvDisplacements`: An array of displacements specifying the starting index in `recvBuffer` for each process.
  communicator.variableAllToAll(
    addr sendBuffer[0],
    addr recvBuffer[0],
    sendCounts,
    recvCounts,
    sendDisplacements,
    recvDisplacements,
    T, T
  )

#[ reduction ]#

proc reduce*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: pointer; 
  recvBuffer: pointer; 
  count: int; 
  datatype: typedesc[T]; 
  root: int
) =
  ## Reduces data from all processes in `communicator` to the process with rank `root` using `MPI_Reduce`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for reduction.
  ##  - `op`: The reduction operation to apply (e.g., `ReduceSum`, `ReduceMaximum`, etc.).
  ##  - `sendBuffer`: A pointer to the data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: A pointer to the buffer to receive the reduced result into. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `count`: The number of elements to send from each process.
  ##  - `datatype`: The MPI datatype of the elements to send and receive.
  ##  - `root`: The rank of the root process that will receive the reduced result.
  mpiCheck MPI_Reduce(
    sendBuffer, 
    recvBuffer, 
    cint(count), 
    mpiType(T), 
    op.mpiOp, 
    cint(root), 
    communicator.comm
  )

proc reduce*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T];  
  root: int;
  sendLen, recvLen: int
) =
  ## Reduces data from all processes in `communicator` to the process with rank `root` using `MPI_Reduce`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for reduction.
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the reduced result into. Only significant at the root process. Can be any type that has a corresponding MPI datatype.
  ##  - `op`: The reduction operation to apply (e.g., `ReduceSum`, `ReduceMaximum`, etc.).
  ##  - `root`: The rank of the root process that will receive the reduced result.
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process (significant at root).
  communicator.reduce(
    op,
    addr sendBuffer[0], 
    addr recvBuffer[0], 
    sendLen, 
    T, 
    root
  )

proc allReduce*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: pointer; 
  recvBuffer: pointer; 
  count: int; 
  datatype: typedesc[T]
) =
  ## Reduces data from all processes in `communicator` and distributes the reduced result to all processes using `MPI_Allreduce`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for reduction.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: A pointer to the data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: A pointer to the buffer to receive the reduced result into. Can be any type that has a corresponding MPI datatype.
  ##  - `count`: The number of elements to send from each process.
  ##  - `datatype`: The MPI datatype of the elements to send and receive.
  mpiCheck MPI_Allreduce(
    sendBuffer, 
    recvBuffer, 
    cint(count), 
    mpiType(T), 
    op.mpiOp, 
    communicator.comm
  )

proc allReduce*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T];  
  sendLen, recvLen: int
) =
  ## Reduces data from all processes in `communicator` and distributes the reduced result to all processes using `MPI_Allreduce`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for reduction.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the reduced result into. Can be any type that has a corresponding MPI datatype.
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process (significant at root).
  communicator.allReduce(
    op,
    addr sendBuffer[0], 
    addr recvBuffer[0], 
    sendLen, 
    T
  )

proc inclusiveScan*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: pointer; 
  recvBuffer: pointer; 
  count: int; 
  datatype: typedesc[T]
) =
  ## Performs an inclusive scan (prefix reduction) across all processes in `communicator` using `MPI_Scan`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: A pointer to the data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: A pointer to the buffer to receive the scanned result into. Can be any type that has a corresponding MPI datatype.
  ##  - `count`: The number of elements to send from each process.
  ##  - `datatype`: The MPI datatype of the elements to send and receive.
  mpiCheck MPI_Scan(
    sendBuffer, 
    recvBuffer, 
    cint(count), 
    mpiType(T), 
    op.mpiOp, 
    communicator.comm
  )

proc inclusiveScan*[T](
  communicator: MpiCommunicator; 
  op: MpiOperation; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T];  
  sendLen, recvLen: int
) =
  ## Performs an inclusive scan (prefix reduction) across all processes in `communicator` using `MPI_Scan`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the scanned result into. Can be any type that has a corresponding MPI datatype.
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process (significant at root).
  communicator.inclusiveScan(
    op,
    addr sendBuffer[0], 
    addr recvBuffer[0], 
    sendLen, 
    T
  )

proc exclusiveScan*[T](
  communicator: MpiCommunicator;
  op: MpiOperation; 
  sendBuffer: pointer; 
  recvBuffer: pointer;  
  count: int; 
  datatype: typedesc[T]
) =
  ## Performs an exclusive scan (prefix reduction) across all processes in `communicator` using `MPI_Exscan`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: A pointer to the data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: A pointer to the buffer to receive the scanned result into. Can be any type that has a corresponding MPI datatype.  
  ##  - `count`: The number of elements to send from each process.
  ##  - `datatype`: The MPI datatype of the elements to send and receive.
  mpiCheck MPI_Exscan(
    sendBuffer, 
    recvBuffer, 
    cint(count), 
    mpiType(T), 
    op.mpiOp, 
    communicator.comm
  )

proc exclusiveScan*[T](
  communicator: MpiCommunicator; 
  op: MpiOperation; 
  sendBuffer: openArray[T]; 
  recvBuffer: var openArray[T];  
  sendLen, recvLen: int
) =
  ## Performs an exclusive scan (prefix reduction) across all processes in `communicator` using `MPI_Exscan`.
  ## 
  ## Parameters:
  ##  - `communicator`: The `MpiCommunicator` to use for the operation.
  ##  - `op`: The reduction operation to apply (e.g., sum, max).
  ##  - `sendBuffer`: The data to send. Can be any type that has a corresponding MPI datatype.
  ##  - `recvBuffer`: The buffer to receive the scanned result into. Can be any type that has a corresponding MPI datatype.  
  ##  - `sendLen`: The number of elements to send from each process.
  ##  - `recvLen`: The number of elements received from each process (significant at root).
  communicator.exclusiveScan(
    op,
    addr sendBuffer[0], 
    addr recvBuffer[0], 
    sendLen, 
    T
  )

#[ MPI dispatch wrappers ]#

macro mpi*(routine: untyped): untyped =
  ## MPI dispatch wrapper
  ## 
  ## Encapsulates body of code within an MPI initialization/finalization block.
  ## Can be used in two ways. 
  ## 
  ## The first is as a pragma that wraps a procedure body:
  ## 
  ## ```nim
  ## proc program() {.mpi.} =
  ##   let comm = WorldCommunicator
  ##   echo "Hello from process ", comm.myRank, " of ", comm.size
  ## program()
  ## ```
  ## 
  ## The second is as a code injection template that wraps arbitrary code:
  ## 
  ## ```nim
  ## mpi:
  ##   let comm = WorldCommunicator
  ##   echo "Hello from process ", comm.myRank, " of ", comm.size
  ## ```
  if routine.kind notin {
    nnkProcDef, 
    nnkFuncDef, 
    nnkMethodDef, 
    nnkIteratorDef, 
    nnkConverterDef
  }: 
    return quote do:
      proc main =
        mpiInit()
        defer: mpiFinalize()
        assert mpiInitialized()
        `routine`
      main()
      assert mpiFinalized()
  else:
    let body = routine[^1]
    result = routine
    routine[^1] = quote do:
      mpiInit()
      defer: mpiFinalize()
      assert mpiInitialized()
      `body`