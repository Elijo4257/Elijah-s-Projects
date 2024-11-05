# Import standard libraries
import os
from glob import glob

# Import scientific computing libraries
import numpy as np
import pandas as pd
from scipy import linalg
import scipy.io

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import audio processing libraries
import librosa
import librosa.display
import soundfile as sf

# Import files from data folder
folder_paths = [] #matrix to store the path to each song so we can easily access it
genres = [] #matrix to catagorize each song so we can give reccomendations later 

datapath = 'data_backup' #this is establishing out folder we wanna reach into 
for entry_name in os.listdir(datapath): #for each name in the directory
    entry_path = os.path.join(datapath, entry_name) # this creates a path to each directory 
    if os.path.isdir(entry_path): #if the directory path exist (meaning its not a .txt or somthing else)
        folder_paths.append(entry_path) # add the  full path to the new matrix we made at the begging to access each directory
        genres.append(entry_name)# add each entry to genre
print(genres)
print(folder_paths)

# Import all song files
files = []  #where we wanna store the song files
for i in range(0,len(folder_paths)): #for every full path in "folder_paths"
    files.append(glob(folder_paths[i]+’/*.wav’)) 
    #so what this is doing is going through each directory and we're checking to see if there is a .wav file in each directory.
    #we check for .wav by using "glob" to find the ".wav" pattern. then if the file is a .wav file we add it to the "files" matrix

# Uses librosa to load the song files into waveform format
waves = [] #stores LISTS of waveform data
sr = [] #stores the sample rates
for i in range(0,len(files)): #for every .wav file in files 
    wavelist = [] # makes an empty list called wavelist
    for j in range(0,len(files[i])): #for everything within that .wav file
        yin, srin = librosa.load(files[i][j],duration=29)
        #librosa.load loads each audiofile (file [i][j]) and only loads the first 29 seconds of that song
        #yin is the audio  time series (waveform) of the file
        #srin is the sample rate of the loaded file
        wavelist.append(yin) # adding the wave form to the wavelist
    waves.append(wavelist) #after processing all the waveforms into a lsit it adds that to waves then goes back and clears the
    #wavelist for the next file, file[i]

# Uses librosa to create Mel-Frequency cepstral coefficient matrices for each song
spectrums = [] #stores a list of list of the spectrum data
for i in range(0,len(waves)): #for each list of waveform in waves 
    spectrumlist = [] # initialize the spectrumlist to be empty 
    for j in range(0,len(waves[i])): #for each waveform in the current list of waveforms 
        specin = librosa.feature.mfcc(y=waves[i][j]) 
        #use the mfcc to reduces the dimensionality of the data 
        #save that low dim representation to specin
        spectrumlist.append(specin) #add the new low dip rep to the sectrum list 
    spectrums.append(spectrumlist) #and the new list to the list spectrums to get a list of list

# Randomly selects songs in each genre based on trainSetSize and splits in to training and test sets
trainSetSize = 30 #set train set size equal to 30
trainIndex = np.random.choice(range(98), trainSetSize, replace=
    False) #trainIndex = training data
    #this randomly selects 30 pieces of data from range 0 - 98
testIndex = np.arange(0,98,1) #creates an array of indices 0 - 98  
testIndex = list(filter(lambda x: x not in trainIndex, testIndex) 
    #now that i have an array 0 -98 this line is filtering out all the training indices so all i get is information not used yet
    #that way my training data and my testing data are different and theres no overlap
    )

testSet = np.take(files[0],testIndex)
#creates a testing set of data from files at indicie testIndex, we do this bc we dont want our training data to overlap
trainSet = np.take(files[0],trainIndex)
#creates a trianing set of data from files at indicie trainIndex,  we do this bc we dont want our training data to overlap

#extracts training and set data from the first genre of files

for i in trainIndex:
    print(i)

averageSongs = [] #this list stores the average MFCC for each genre
averageSpectrum = [] #this stores the average MFCC across all genres
iterations = 0
for i in range(0,len(genres)):  #for each genre
    average = [] #Initialize an empty list to store the average MFCCs for the current genre
    for j in trainIndex: #for each entry in the trianing data
        if len(average) == 0: #if the average array is empty
            average = spectrums[i][j] # if empty this assigns average to the current MFCC spectrum
        else:
            average = average + spectrums[i][j] #if its not empty add the MFCC spectrum data to the average list
        if len(averageSpectrum) == 0:  #if the averageSpectrun list is empty 
            averageSpectrum = spectrums[i][j] #averageSepctrum gets assigned to the current speactrum
        else:
            averageSpectrum = averageSpectrum + spectrums[i][j] #add the spectrum data to the list
        iterations = iterations + 1 
    averageSongs.append(average*(1/trainSetSize)) 
    #this computes the average MFCC for each genre by dividing the average by the trainSetSize and adds it to averageSongs 
averageSpectrum = averageSpectrum*(1/(trainSetSize*len(genres)))
#this comnputes the average MFCC for all genres (key here is the multiplication by the len(genres))

# Creates a training basis for all songs in the training set for each genre in order.
# This will also be mean-centered
trainBases = []
for i in range(0,len(spectrums)): #for each entry in the spectrum data
    first = 1 #intializing
    for j in trainIndex: #for each entry in the training data
        if bool(first) == True: # checks to see if this is the first iteration 
            X = (spectrums[i][j]-averageSpectrum) # subtracts average spectrum data from the one were currently looking at to create
            #highly specialized spectrum data for each song
            first = 0 # updates "first"
        else: #if not the first iteration 
            X = np.concatenate((X,(spectrums[i][j]-averageSpectrum
                )),axis=1) # for genres that have more than one song this make each song gets the average song subtracted from it 
    trainBases.append(X) #add all the new(unique) songs to the training base
Songs = []
for i in range(0,len(trainBases)): #for each genre in training
    if len(Songs) == 0: #if there is no songs in the genre
        Songs = trainBases[i] #initalize with first training basis
    else:# if not empty 
        Songs = np.append(Songs,trainBases[i],axis=1)  #add the training bases to the "Songs" basis
# Performs SVD on the basis matrix

#this is where i can make my change and use the randomized SVD
U,Sigma,V = np.linalg.svd(Songs,full_matrices=0) #find the SVD


#performs PCA by projecting the average mfcc matrix of each genre onto the selected columns of U transpose.
modes = [1,2]  #contains the mode indicices to project onto 
coords = []#stores the projected coordinates of the average MFCC's

for i in range(0,len(averageSongs)): #for each genres average MFCC
#len(averageSongs) = numb of genres
    coords.append(U[:,modes-np.ones_like(modes)].T @ averageSongs[i])
    #this subtracts 1 from every element in modes, coverting [1,2] or [0,1]
    #then takes the columns of U specified to the modes (transposed)
    #multiply this by the current genre to get the genre porjected onto specified modes
    #add the projected coordinates to "coords"

for i in range(0,len(coords)): #goes through the list of the frst list of coordinates
    plt.scatter(coords[i][0,:],coords[i][1,:],label=genres[i]) 
    #coords[i][0,:] = x-coordinate (projecttion onto mode 1)
    #coords[i][1,:] = y-coordinate (projecttion onto mode 2)

#here im graphing
plt.xlabel("{}{}".format("Mode ", modes[0]))
plt.ylabel("{}{}".format("Mode ", modes[1]))
plt.legend()
plt.show()

# Performs classification by projecting each song in the test set and assigning a classificaiton based on the distance to each
# PCA cluster.
testClassif = [] #stores predicted classification for each genre

for k in range(0,len(genres)): #for each genre
    predictedValues = [] 
    print(genres[k]) #print current genre
    for j in testIndex: #for each element in the testing data
        testcoord = U[:,modes-np.ones_like(modes)].T @ (spectrums
            [k][j]) #testcoord is the spectrum MFCC of genre k and index j projected onto the modes we decided at 
        norms = [] #distance list
        for i in range(0,len(genres)): #for each genre
            norms.append(np.linalg.norm(testcoord-coords[i],ord=’
                nuc’)) #this finds the nuclear norm of the distance between the test coordinate and the average coordinate of genre i
        predictedValues.append(np.argmin(norms)) #add the genre with the min norm (closest to) predicted values
    
    testClassif.append(predictedValues) #adds tp predicted genre

# Calculates the correct classification rates for each genre.

classifRates = [] #stores clasification accuracy rates for each classification
for i in range(0,len(testClassif)): #for each genre
    sumCorrect = 0
    for k in range(0,len(testClassif[i])): #for the current genre
        if testClassif[i][k] == i: #checks if the predicted genre == the true genre
            sumCorrect = sumCorrect + 1 #increments
    classifRates.append(sumCorrect/len(testClassif[i]))#gets accurracy rate

classifRates

