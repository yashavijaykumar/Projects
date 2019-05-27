# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 18:05:59 2017

@author: Yasha v
"""
from __future__ import print_function

import sys # provides information about constants, functions and methods of the Python interpreter
import random #contains a variety of things to do with random number generation
from multiprocessing import Queue 



def generateMaze(n):
    if(n < 5 or n > 10):
       
        newInput = int(input("Rook Jumping Maze size (5-10)?"));
       
        generateMaze(newInput);
        sys.exit()
    else:
       
        generateJumpingMaze(n);
	        
 

def generateJumpingMaze(size):
	
    arr = []
   
    static = []
    
    for num in range(size):
        arr.append([])
        static.append([])
  
    for x in range (size):
        for y in range(size):
          
            maximums = [0,0,0,0];
            
            rmax = 0;
            
            cmax = 0;
            
            rtracker = x+1;
            
            ctracker = y+1;
           
            right_diff = size - ctracker;
            left_diff = ctracker - 1;
            above_diff = size - rtracker;
            below_diff = rtracker - 1;
           
            maximums[0] = right_diff;
            maximums[1] = left_diff;
            maximums[2] = above_diff;
            maximums[3] = below_diff;
           
            maxMove = max(maximums);
           
            randomMax = random.randint(1, maxMove);
            if(x == size-1 and y == size-1):
                goal = 0;
                arr[x].append(goal);
                static[x].append(goal);
            else:
                arr[x].append(randomMax);
                static[x].append(randomMax);
    print("generated jumping maze");
    printMatrix(arr, size);
  
    return static, bfsCheck(0,0,arr,size);


def printMatrix(arr, size):
    for x in range(size):
        print(arr[x]);


def GetPathFromNodes(node, originalMatrix): 
    path = [] 
    finalpath = []
    formatt = []
   
    while(node != None): 
    	
        path.append([node[0],node[1], node[2]])
      
        node = node[3] 
   
    finalpath = list(reversed(path)) 
  
    i = 0
    while i < len(finalpath)-1:
    	if(finalpath[i+1] is None):
    		break;
    	
    	if (finalpath[i][0] < finalpath[i+1][0]):
    		formatt.append('D')
    	
    	if(finalpath[i][0] > finalpath[i+1][0]):
    		formatt.append('U')
    	
    	if(finalpath[i][1] < finalpath[i+1][1]):
    		formatt.append('R')

    	if(finalpath[i][1] > finalpath[i+1][1]):
    		formatt.append('L')
    	
    	i = i + 1
    
    print(formatt)
    energy = -len(formatt)
    print('Out total energy is: ' + str(energy))
    return energy


def bfsCheck(x,y,Matrix,size):
  
    q = Queue()
    
    energy = 0
   
    step = 0
    originalMatrix = list(Matrix)
     
    matrixpos = originalMatrix[x][y];
   
    print("The root position value: " + str(matrixpos));
  
    q.put([x,y, matrixpos, None])
    while(q.qsize() > 0):
     
        node = q.get();
        x = node[0];
        y = node[1];
        dist = node[2];
     
        goal=0;
        if (originalMatrix[x][y] == goal):
        	
        	return GetPathFromNodes(node, originalMatrix)
           
        #Set this index to P now, so that we know that we have explored it, P for player I guess
        originalMatrix[x][y] = 'P'
       
        #If the current node we are at is already labeled P, then continue to next iteration
        if(dist == 'P'):
        	
        	continue
        #Number of steps
        step = step + 1;
        
        for i in [[x-dist,y], [x+dist,y], [x,y-dist], [x,y+dist]]:
            
            firstpos = i[0];
            secondpos = i[1];
           
            if((size > firstpos >= 0) and (size > secondpos >= 0)):
                newPos = originalMatrix[firstpos][secondpos];
               
                q.put([firstpos,secondpos,newPos,node])
                
        energy = energy + step

    energy = energy*10*-1
    print('Energy: ' + str(energy))
        
    return energy
    
    

def randomGen(iterations):
	
	constant, energy = generateJumpingMaze(5)
	printMatrix(constant, 5);
	print(energy)
	counter = 0
	temp = list(constant)
	q = 0;
	print("THis is TEMP initiallly>>>>>>>>>>>>>>>:")
	printMatrix(temp,5)
	while(counter < iterations):
		randomx = random.randint(0,4);
		randomy = random.randint(0,4);
		if(randomx == 4 and randomy == 4):
			randomy = random.randint(0,3);
		
		currentval = constant[randomx][randomy];
		maximums = [0,0,0,0];
	    	
		rtracker = randomx+1
	    	
		ctracker = randomy+1;
	    	
		right_diff = 5 - ctracker;
		left_diff = ctracker - 1;
		above_diff = 5 - rtracker;
		below_diff = rtracker - 1;
        	
		maximums[0] = right_diff;
		maximums[1] = left_diff;
		maximums[2] = above_diff;
		maximums[3] = below_diff;
    	
		maxMove = max(maximums);
    	
		randomMax = random.randint(1, maxMove);
		print("Current Jump Number: " + str(currentval) + ' at Position [' + str(randomx) + '][' + str(randomy) + ']')
		print("New Random Legal number: " +str(randomMax) + ' at Position [' + str(randomx) + '][' + str(randomy) + ']')
		while(currentval == randomMax):			randomMax = random.randint(1, maxMove)
		print("Random Legal number in loop: " +str(randomMax))
		constant[randomx][randomy] = randomMax;
		print("This is our initial edited matrix here");
		printMatrix(constant,5)
		
		newenergy = bfsCheck(0,0,constant,5)
		
		print("Our energy before is " + str(energy))
		print("Our new energy is: " + str(newenergy))
		if(newenergy>energy):
			print("New Energy is better")
			energy = newenergy
			temp[randomx][randomy] = randomMax;
			constant = list(temp);
			print("Printing Better Matrix:")
			printMatrix(temp, 5)
			
		if(energy>=newenergy):
			print("Keeping old matrix")
			constant = list(temp);
		
		counter = counter + 1;
		return print(energy)


def generatePremadeMaze(arr, size):
	static = list(arr)
	return static, bfsCheck(0,0,static, size)




randomGen(1)