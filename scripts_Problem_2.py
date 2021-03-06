#Birthday Cake Candles

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    max=candles[0]
    count=0
    for i in candles:
        if i==max:
            count+=1
        elif i>max:
            count=1
            max=i
    return count

if __name__ == '__main__':

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    print(str(result) + '\n')


#Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    if v1==v2:
        return 'NO'
    
    
        
    ds=x2-x1
    dv=v2-v1
    #if j is the number of jumps, then they must meet at j=-ds/dv, so they meet iff ds%dv==0
        
    if ds*dv>0:
        return('NO')
    
    ds=abs(ds)
    dv=abs(dv)
    
    if ds%dv==0:
        return 'YES'
    
    return 'NO'

if __name__ == '__main__':

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    print(result + '\n')

#Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys


def viralAdvertising(n):
    L=2
    C=2

    for _ in range(n-1):
        L*=3
        L=math.floor(L/2)
        C+=L
        
    return C

if __name__ == '__main__':

    n = int(input().strip())

    result = viralAdvertising(n)

    print(str(result) + '\n')


#Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    #it's easy to see that superDigit(n,k)=superDigit(k*superDigit(n,1),1)
    
    def singlesuperDigit(n):
        while len(n)>1:
            temp=0
            for i in n:
                temp+=int(i)
            n=str(temp)
        return int(n)
    
    return(singlesuperDigit(str(k*singlesuperDigit(n))))

if __name__ == '__main__':

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    print(str(result) + '\n')


#Insertion Sort - Part 1

import math
import os
import random
import re
import sys


def insertionSort1(n, arr):
    ins=arr[-1]
    i=n-2
    
    while i>=0 and ins<arr[i]:
        arr[i+1]=arr[i]
        i-=1
        print(*arr)
    
    arr[i+1]=ins
    print(*arr)
            
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

    


#Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    ins=arr[n-1]
    i=n-2
    
    while i>=0 and ins<arr[i]:
        arr[i+1]=arr[i]
        i-=1
    
    arr[i+1]=ins


def insertionSort2(n, arr):
    for i in range(2,n+1):
        insertionSort1(i,arr)
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
