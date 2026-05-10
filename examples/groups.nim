import nimpi

mpi:
  assert WorldCommunicator.size >= 4, "This example requires at least 4 processes"

  # create a new group from a sequence of ranks
  var customGroup1 = newMpiGroup(@[0, 2])
  var customGroup2 = newMpiGroup(@[0, 1, 3])

  # create a new communicator from the custom group
  var customGroup3 = customGroup1 + customGroup2 # full set of processes
  var customGroup4 = customGroup1 * customGroup2 # process 0
  var customGroup5 = customGroup1 - customGroup2 # just group

  # create communicators from the groups
  var customComm3 = customGroup3.newMpiCommunicator()
  var customComm4 = customGroup4.newMpiCommunicator()
  var customComm5 = customGroup5.newMpiCommunicator()