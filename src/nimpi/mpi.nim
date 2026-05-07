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
## High-level, idiomatic Nim bindings for MPI (Message Passing Interface).
##
## Quick Start
## ===========
##
## ```nim
## import nimpi
##
## mpi:
##   echo "Rank: ", myRank(WorldCommunicator)
##   echo "Size: ", size(WorldCommunicator)
## ```
##
## Core Types
## ==========
##
## - `MpiCommunicator`: Wraps MPI_Comm for safe group communication.
## - `MpiError`: Exception raised when MPI operations fail.
##
## Communicator Operations
## =======================
##
## Split, duplicate, and manage MPI communicators:
##
## - `split()`: Create subcommunicators by color/rank.
## - `duplicate()`: Clone a communicator.
## - `size()`, `myRank()`: Query communicator metadata.
## - `free()`: Release communicator resources.
##
## Initialization & Finalization
## ==============================
##
## Use the `mpi` template to safely initialize/finalize MPI:
##
## ```nim
## mpi:
##   # MPI_Init and MPI_Finalize called automatically
##   discard
## ```

import mpiwrap

#[ MPI types ]#

type MpiError* = object of CatchableError
  ## Represents an error that occurs during MPI operations. Contains an error code
  ## that can be used to identify the specific MPI error that occurred. Paired with
  ## the `mpiAssert` procedure, which raises an `MpiError` when an MPI function returns
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

#[ global variables ]#

let
  WorldCommunicator* = MpiCommunicator(comm: MPI_COMM_WORLD)
    ## The default communicator that includes all processes in the MPI program.
    
  SelfCommunicator* = MpiCommunicator(comm: MPI_COMM_SELF)
    ## A communicator that includes only the calling process. 

#[ initialization/finalization ]#

proc mpiInit* = 
  ## Initializes the MPI environment. Must be called before any other MPI functions.
  mpiAssert MPI_Init(nil, nil)

proc mpiFinalize* = 
  ## Finalizes the MPI environment. Should be called after all MPI functions.
  mpiAssert MPI_Finalize()

proc mpiInitialized*: bool = 
  ## Checks if the MPI environment has been initialized.
  var flag: cint
  mpiAssert MPI_Initialized(addr flag)
  return flag != 0

proc mpiFinalized*: bool = 
  ## Checks if the MPI environment has been finalized.
  var flag: cint
  mpiAssert MPI_Finalized(addr flag)
  return flag != 0

#[ error handling ]#

proc mpiAssert*(code: cint) =
  ## Asserts that an MPI function call was successful. If the code is not `MPI_SUCCESS`,
  ## raises an `MpiError` with the corresponding error message.
  ## 
  ## Parameters:
  ##  - `code`: The return code from an MPI function call to check for success.
  ## 
  ## Example:
  ## ```nim
  ## var rank: cint
  ## mpiAssert MPI_Comm_rank(WorldCommunicator.comm, addr rank)
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

#[ communicator ]#

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
  mpiAssert MPI_comm_dup(communicator.comm, addr comm)
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
  mpiAssert MPI_Comm_split(communicator.comm, cint(color), cint(key), addr comm)
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
  mpiAssert MPI_Abort(communicator.comm, cint(errorcode))

proc free*(communicator: var MpiCommunicator) =
  mpiAssert MPI_Comm_free(addr communicator.comm)

proc size*(communicator: MpiCommunicator): int =
  var size: cint
  mpiAssert MPI_Comm_size(communicator.comm, addr size)
  return int(size)

proc myRank*(communicator: MpiCommunicator): int =
  var rank: cint
  mpiAssert MPI_Comm_rank(communicator.comm, addr rank)
  return int(rank)

#[ MPI block ]#

template mpi*(body: untyped): untyped =
  ## Encapsulates MPI program in a block that ensures proper initialization and 
  ## finalization of MPI. Alternative to manual calls to `mpiInit` and `mpiFinalize`.
  ## 
  ## Example:
  ## ```nim
  ## mpi:
  ##   let comm = WorldCommunicator
  ##   echo "Hello from process ", comm.myRank, " of ", comm.size
  ## ```
  proc main =
    mpiAssert MPI_Init(nil, nil)
    defer: mpiAssert MPI_Finalize()
    assert mpiInitialized()
    body
  main()
  assert mpiFinalized()