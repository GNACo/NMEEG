import synapseclient 
import synapseutils 

syn = synapseclient.Synapse()
authToken = input("Enter your Synapse token: ") 
syn.login(authToken=authToken) 
files = synapseutils.syncFromSynapse(syn, 'syn53497784') 
 