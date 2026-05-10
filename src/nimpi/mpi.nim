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
  MpiCommunicator* = object
    ## Represents an MPI communicator, which is a group of processes that can 
    ## communicate with each other. MPI communicators are fundamental to MPI 
    ## programming, as they define the context for communication operations.
    ## 
    ## Attributes:
    ## - `comm`: The underlying MPI_Comm handle that represents the communicator.
    comm*: MPI_Comm
  
  MpiGroup* = object
    ## Represents an MPI group, which is an ordered set of processes. MPI groups
    ## are used to define the membership of communicators and to specify subsets
    ## of processes for communication operations.
    ##
    ## Attributes:
    ## - `group`: The underlying MPI_Group handle that represents the group.
    group*: mpiwrap.MPI_Group

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
  for i in 0..<ranks.len:
    cRanks[i] = cint(ranks[i])
  mpiCheck MPI_Group_incl(WorldCommunicator.group.group, cint(ranks.len), addr cRanks[0], addr group)
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
  for i in 0..<ranks.len:
    cRanks[i] = cint(ranks[i])
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

proc size*(communicator: MpiCommunicator): int =
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

proc myRank*(communicator: MpiCommunicator): int =
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
    if `communicator`.myRank == 0: echo `message`

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