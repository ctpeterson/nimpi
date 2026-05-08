import nimpi

mpi:
  # duplicate the world communicator
  var customComm1 = WorldCommunicator.duplicate()

  # echo a message from the custom communicator
  customComm1.echo "Hello from customComm1; # ranks = ", customComm1.size

  # split the world communicator into two sub-communicators 
  var customComm2 = WorldCommunicator.split(
    WorldCommunicator.myRank mod 2, 
    WorldCommunicator.myRank
  )

  # echo a message from the custom communicator
  customComm2.echo "Hello from customComm2; # ranks = ", customComm2.size

  # no need to manually free communicators; they will be freed when they go 
  # out of scope