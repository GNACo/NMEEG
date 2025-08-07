import synapseclient 
import synapseutils 
 
syn = synapseclient.Synapse()
authToken = input("Enter your Synapse token: ")
syn.login(authToken="") 
files = synapseutils.syncFromSynapse(syn, 'syn53497914') 