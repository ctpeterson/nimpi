import nimpi

mpi:
  let rank = WorldCommunicator.myRank
  let size = WorldCommunicator.size
  echo "Hello, world! From: ", rank, "/", size

# alternative (1):
# mpiInit()
# let rank = WorldCommunicator.myRank
# let size = WorldCommunicator.size
# echo "Hello, world! From: ", rank, "/", size
# mpiFinalize()

# alternative (2):
# proc program {.mpi.} =
#   let rank = WorldCommunicator.myRank
#   let size = WorldCommunicator.size
#   echo "Hello, world! From: ", rank, "/", size
# program()