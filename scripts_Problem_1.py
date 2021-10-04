#Set .union() Operation

input()
eng=set(input().split())
input()
fr=set(input().split())

print(len(eng.union(fr)))

#Set .intersection() Operation

input()
eng=set(input().split())
input()
fr=set(input().split())

print(len(eng.intersection(fr)))

#Set .difference() Operation

input()
eng=set(input().split())
input()
fr=set(input().split())

print(len(eng.difference(fr)))

#Set .symmetric_difference() Operation

input()
eng=set(input().split())
input()
fr=set(input().split())

print(len(eng.symmetric_difference(fr)))

#Set Mutations

input()
A=set(map(int,input().split()))
N=int(input())
for _ in range(N):
    [command,null]=input().split()
    B=set(map(int,input().split()))

    getattr(A,command)(B)

print(sum(A))

#Check Subset

N=int(input())

for _ in range(N):
    input()
    A=set(input().split())
    input()
    B=set(input().split())
    
    print(A.issubset(B))

#Check Strict Superset

A=set(input().split())
N=int(input())

result=True
for _ in range(N):
    B=set(input().split())
    
    result=result and B.issubset(A) and A!=B
print(result)

#The Captain's Room

K=int(input())
num=input().split()
s=set()
n=set()

for i in num:
    if i in s:
        n.add(i)
    else:
        s.add(i)
    
print((s.difference(n)).pop())

#collections.Counter()

from collections import Counter

input()
sizes=Counter(input().split())
C=int(input())

money=0
for i in range(C):
    [s,price]=input().split()
    if sizes[s]>0:
        money+=int(price)
        sizes[s]-=1
        
print(money)


#DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)

[n,m]=map(int,input().split())

for i in range(n):
    d[input()].append(str(i+1))
    
for i in range(m):
    temp=d[input()]
    if len(temp)>0:
        print(' '.join(temp))
    else:
        print(-1)


#Collections.namedtuple()

from collections import namedtuple
N=int(input())
students=namedtuple('students', input())
print("{:.2f}".format(sum(map(int,[(students._make(input().split())).MARKS for i in range(N)]))/N))

#Collections.OrderedDict()

from collections import OrderedDict
od=OrderedDict()
N=int(input())
for _ in range(N):
    s=input()
    i=len(s)-1
    while i>=0:
        if s[i]==' ':
            idx=i
            i=0
        i-=1
    name=s[0:idx]
    price=int(s[idx+1:])
    if name in od.keys():
        od[name]+=price
    else:
        od[name]=price

for i in od.keys():
    print(i+' '+str(od[i]))

#Word Order

from collections import OrderedDict

od=OrderedDict()
N=int(input())
for _ in range(N):
    word=input()
    if word in od.keys():
        od[word]+=1
    else:
        od[word]=1

print(len(od.keys()))
print(' '.join([str(od[i]) for i in od.keys()]))

#Collections.deque()

from collections import deque

N=int(input())
d = deque()

for _ in range(N):
    s=input().split()
    comm=s[0]
    if len(s)>1:
        arg=int(s[1])
        getattr(d,comm)(arg)
    else:
        getattr(d,comm)()
    

print(' '.join([str(i) for i in list(d)]))

#Company Logo

import math
import os
import random
import re
import sys
from collections import OrderedDict


if __name__ == '__main__':
    s = input()

    alpha=[]
    for i in range(26):
        alpha.append(chr(ord('a')+i))

    od=OrderedDict()

    for c in alpha:
        od[c]=0
    
    for c in s:
        od[c]+=1
        
        
    commons=[]
    val=list(od.values())

    for _ in range(3):
        m=max(val)
        idx=val.index(m)
        commons.append([idx,m])
        val[idx]=-1
    
    for i in commons:
        print(alpha[i[0]]+' '+str(i[1]))


#Piling Up!

from collections import deque

    #a list verifies then property only if the max is at the edge and the list without that edge verifies the property

def iscube(cube,length):
    if length>2:
        r=cube.pop()
        l=cube.popleft()
        if r>l:
            length-=1
            cube.appendleft(l)
            old=r
        elif l>r:
            length-=1
            cube.append(r)
            old=l
        elif r==l:
            #if the max is at both edges we can remove both in an arbitrary order (but we must remove both before passing to another element)
            length-=2
            old=r
    else:
        return True
        
        
    while length>2:
        r=cube.pop()
        l=cube.popleft()
        if max([r,l])>old:
            return False
        if r>l:
            length-=1
            cube.appendleft(l)
            old=r
        elif l>r:
            length-=1
            cube.append(r)
            old=l
        elif r==l:
            #if the max is at both edges we can remove both in an arbitrary order (but we must remove both before passing to another element)
            length-=2
            old=r
            
    if length==2:
        r=cube.pop()
        l=cube.popleft()
        m=max([r,l])
        
    elif length==1:
        r=cube.pop()
        m=r
    
    elif length==0:
        return True
    
    if old>=m:
        return True
    else:
        return False


N=int(input())

for _ in range(N):
    length=int(input())
    cube=map(int,input().split())
    cube=deque(cube)
    if iscube(cube,length):
        print('Yes')
    else:
        print('No')

#Calendar Module

import calendar

s=list(map(int,input().split()))

day=calendar.weekday(s[2],s[0],s[1])

if day==0:
    print('MONDAY')
elif day==1:
    print('TUESDAY')
elif day==2:
    print('WEDNESDAY')
elif day==3:
    print('THURSDAY')
elif day==4:
    print('FRIDAY')
elif day==5:
    print('SATURDAY')
elif day==6:
    print('SUNDAY')

#Time Delta

import calendar
import math
import os
import random
import re
import sys

def time_delta(t1, t2):
    
    #dictionary to convert automatically months to numbers
    #january:1-----december:12
    
    months={}
    j=0
    for i in list(calendar.month_abbr):
        months[i]=j
        j+=1
    
    t1=t1.split()
    t2=t2.split()
    
    date1=[int(t1[1]),months[t1[2]],int(t1[3])]
    date2=[int(t2[1]),months[t2[2]],int(t2[3])]
    
    time1=list(map(int,t1[4].split(':')))
    time2=list(map(int,t2[4].split(':')))
    
    gmt1=int(t1[5])
    if gmt1>=0:
        gmt1=60*60*(gmt1//100)+60*(gmt1%100)
    else:
        gmt1=-(60*60*(abs(gmt1)//100)+60*(abs(gmt1)%100))
    gmt2=int(t2[5])
    if gmt2>=0:
        gmt2=60*60*(gmt2//100)+60*(gmt2%100)
    else:
        gmt2=-(60*60*(abs(gmt2)//100)+60*(abs(gmt2)%100))
        
    startyear1=0
    for i in range(date1[1]-1):
        startyear1+=calendar.monthrange(date1[2],i+1)[1]*24*60*60
    startyear2=0
    for i in range(date2[1]-1):
        startyear2+=calendar.monthrange(date2[2],i+1)[1]*24*60*60
        
    startyear1+=(date1[0]-1)*24*60*60
    startyear2+=(date2[0]-1)*24*60*60
    
    startyear1-=gmt1
    startyear2-=gmt2
    
    startyear1+=time1[0]*60*60+time1[1]*60+time1[2]
    startyear2+=time2[0]*60*60+time2[1]*60+time2[2]
    
    
    if date1[2]>=date2[2]:
        print(abs(calendar.leapdays(date2[2],date1[2])*24*60*60+(date1[2]-date2[2])*365*24*60*60+startyear1-startyear2))
    else:
        print(calendar.leapdays(date1[2],date2[2])*24*60*60+(date2[2]-date1[2])*365*24*60*60+startyear2-startyear1)
    
if __name__ == '__main__':
    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

#Exceptions

N=int(input())

for _ in range(N):
    s=input().split()
    
    try:
        a=int(s[0])
        b=int(s[1])
        c=a//b
        print(c)
    except ValueError as e:
        print("Error Code:",e)
    except ZeroDivisionError as e:
        print("Error Code:",e)

#Zipped!

[N,X]=list(map(int,input().split()))

sub=[]

for _ in range(X):
    sub.append(list(map(float,input().split())))

stud=list(zip(*sub))

for i in stud:
    print('{:.1f}'.format(sum(i)/X))

#Athlete Sort

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    

    arr.sort(key=lambda x: x[k])
    
    for i in arr:
        print(' '.join(map(str,i)))

#ginortS

def custom_key(x):
    d={}
    
    for i in range(26):
        d[chr(ord('a')+i)]=i
    
    for i in range(26):
        d[chr(ord('A')+i)]=i+26
        
    for i in range(5):
        d[str(2*i+1)]=52+i
    
    for i in range(5):
        d[str(2*i)]=57+i
        
    return(d[x])

s=input()

print(''.join(sorted(s,key=custom_key)))

#Map and Lambda Function

cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    if n<=2:
        return([x for x in range(n)])
    l=[0,1]
    for i in range(n-2):
        l.append(l[i]+l[i+1])
    
    return(l)

#Detect Floating Point Number

import re

N=int(input())

p=re.compile(r'^[+-]?\d*\.\d+$')
#^ serve a dire che l'espressione e' all'inizio della stringa, non necessario visto che ho usato match()
#[+-] indica che deve esserci un carattere tra + e -
# (e il ? dopo [+-] indica che va bene anche nessuno dei due)
#\d+ indica che possono esserci 0 o piu' cifre qui
#\. indica che qui ci va un .
#\d+ indica che deve esserci ALMENO una cifra qui
#$ indica che la stringa finisce qui (o che c'e' un carattere newline qui)

for _ in range(N):
    s=input()
    m=p.match(s)  #match controlla solo l'inizio della stringa, search controlla anche in mezzo
    res=bool(m)
    try:
        float(s)
    except:
        res=False
    print(res)

#Re.split()

regex_pattern = r"[,.]+"

#Group(), Groups() & Groupdict()

import re

p=re.compile(r'^.*([a-zA-Z0-9])\1.*$') #\w are alphanumeric characters AND _

s=input()

s=s[::-1] #reverses the string, because group(1) gives the content of the last group that matched
m=p.match(s)

if bool(m):
    print(m.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()

import re

p=re.compile(r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])') #(?=...) is a positive lookahead assertion, (?<=...) is a positive lookbehind assertion

s=input()

m=p.findall(s)

if bool(m):
    for i in m:
        print(i)
else:
    print(-1)

#Re.start() & Re.end()

import re

s=input()
p=input()
p='(?=('+p+'))' #the lookahead is zero-width so the match will not be overlapping, but the capturing group will still store the match
p=re.compile(p)


m=list(p.finditer(s))

if m:
    for i in m:
        print((i.start(1),i.end(1)-1)) #the end of the match is after the character, the index of the last character is end()-1
else:
    print((-1,-1))

#Regex Substitution

import re

p1=re.compile('(?<=\s)&&(?=\s)')
p2=re.compile('(?<=\s)\|\|(?=\s)')

N=int(input())

text=''

for _ in range(N):
    text=text+input()+'\n'

text=p1.sub('and',text)

text=p2.sub('or',text)

print(text)

#Validating Roman Numerals

regex_pattern = r"M{0,3}(C[MD]|D?C{0,3})(X[LC]|L?X{0,3})(I[XV]|V?I{0,3})$"	

#M{0,3} gives 0,1000,2000,3000
#(C[MD]|D?C{0,3}) gives either 900 and 400 or 0,100,200,300,500,600,700,800
#(X[LC]|L?X{0,3}) gives either 90 and 40 or 0,10,20,30,50,60,70,80
#(I[XV]|V?I{0,3}) gives either 9 and 4 or 0,1,2,3,5,6,7,8
#$ forces the match until the end of the line

#Validating phone numbers

import re

N=int(input())

p=re.compile(r'[789]\d{9,9}$')

for _ in range(N):
    m=p.match(input())
    if bool(m):
        print('YES')
    else:
        print('NO')

#Validating and Parsing Email Addresses

import email.utils
import re


p=re.compile(r'^.* <[A-Za-z][a-zA-Z0-9_\-.]*@[A-Za-z]+\.[A-Za-z]{1,3}>$')

N=int(input())

for _ in range(N):
    address=input()
    if p.match(address):
        address=email.utils.parseaddr(address)
        print(email.utils.formataddr(address))

#Hex Color Code

import re

p=re.compile('(?<=.)(#[A-Fa-f0-9]{3,3}|#[A-Fa-f0-9]{6,6})(?=[^A-Za-z0-9])')

N=int(input())

i=0
while i<N:
    s=input()
    if s[-1:-2:-1]=='{': #s[-1:-2:-1] gives last element
        while s!='}':
            s=input()
            m=p.finditer(s)
            for j in m:
                print(j.group(1))
            i+=1
    i+=1

#HTML Parser - Part 1

import html.parser


N=int(input())

# create a subclass and override the handler methods
class MyHTMLParser(html.parser.HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs:
            if len(attr)<2:
                attr.append(None)
            print('->',attr[0],'>',attr[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            if len(attr)<2:
                attr.append(None)
            print('->',attr[0],'>',attr[1])
        
        
s=''
for _ in range(N):
    s=s+input()
        
parser = MyHTMLParser()
parser.feed(s)

#HTML Parser - Part 2

from html.parser import HTMLParser
#import html.parser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data.count('\n')>0:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(data)
        
    def handle_data(self, data):
        if data!='\n':
            print('>>> Data')
            print(data)
  
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values

import html.parser


N=int(input())

# create a subclass and override the handler methods
class MyHTMLParser(html.parser.HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print('->',attr[0],'>',attr[1])
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print('->',attr[0],'>',attr[1])
        
        
s=''
for _ in range(N):
    s=s+input()
        
parser = MyHTMLParser()
parser.feed(s)

#Validating UID

N=int(input())

def valid(s):
    if len(s)!=10:
        return False
    if len(set(s))!=10:
        return False
    uppercount=0
    digitcount=0
    for i in s:
        if i.isupper():
            uppercount+=1
        elif i.isdigit():
            digitcount+=1
        else:
            if not i.islower():
                return False
    
    if uppercount<2:
        return False
    if digitcount<3:
        return False
    
    return True

for _ in range(N):
    uid=input()
    if valid(uid):
        print('Valid')
    else:
        print('Invalid')

#Validating Credit Card Numbers

import re

def check(s):
    try:
        assert re.match(r'^([456]\d{3}-\d{4}-\d{4}-\d{4}|[456]\d{15})$',s)
        s=s.replace('-','')
        assert not re.search(r'(\d)\1\1\1',s)
    except AssertionError:
        return False
    else:
        return True

N=int(input())

for _ in range(N):
    s=input()
    print('Valid' if check(s) else 'Invalid')

#Validating Postal Code

regex_integer_in_range = r"^[1-9]\d{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.

#Matrix Script

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

s=''
for j in range(m):
    for i in range(n):
        s+=matrix[i][j]
        
print(re.sub(r'(?<=[A-Za-z0-9])[^A-Za-z0-9]+?(?=[A-Za-z0-9])',' ',s))

#XML 1 - Find the Score

def get_attr_number(node):
    return sum([len(elem.attrib) for elem in node.iter()])

#XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level): #use level to track the level of the recursion
    global maxdepth
    level+=1
    maxdepth=max(maxdepth,level)
    
    temp=list(elem)
    if len(temp)==0:
        return None
    else:
        for child in temp:
            depth(child,level)

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        l=['+91 '+str(x[-10:-5])+' '+str(x[-5:]) for x in l]
        f(l)
    return fun

#Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        return([f(x) for x in people])
    return inner

#Arrays

def arrays(arr):
    return numpy.array(arr[::-1],float)


#Shape and Reshape

import numpy



s=input().split()
s=numpy.array(s,int)
s.shape=(3,3)
print(s)


#Transpose and Flatten

import numpy



s=numpy.array([list(map(int,input().split())) for _ in range(int(input().split()[0]))])

print(numpy.transpose(s))
print(s.flatten())


#Concatenate

import numpy



[N,M,P]=map(int,input().split())

a=numpy.array([input().split() for _ in range(N)],int)
b=numpy.array([input().split() for _ in range(M)],int)

print(numpy.concatenate((a, b), axis = 0))

#Zeros and Ones

import numpy



s=tuple(map(int,input().split()))
print(numpy.zeros(s,dtype = int))
print(numpy.ones(s,dtype = int))

#Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')


s=list(map(int,input().split()))
print(numpy.eye(*s))

#Array Mathematics

import numpy


N=int(input().split()[0])
A=numpy.array([input().split() for _ in range(N)],int)
B=numpy.array([input().split() for _ in range(N)],int)

print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

#Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')

A=numpy.array(input().split(),float)

print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

#Sum and Prod

import numpy

N=int(input().split()[0])
A=numpy.array([input().split() for _ in range(N)],int)

print(numpy.prod(numpy.sum(A, axis = 0)))

#Min and Max

import numpy

N=int(input().split()[0])
A=numpy.array([input().split() for _ in range(N)],int)

print(numpy.max(numpy.min(A, axis = 1)))

#Mean, Var, and Std

import numpy


N=int(input().split()[0])
A=numpy.array([input().split() for _ in range(N)],int)

print(numpy.mean(A, axis = 1))
print(numpy.var(A, axis = 0))
print(round(numpy.std(A), 11))

#Dot and Cross

import numpy

N=int(input())
A=numpy.array([input().split() for _ in range(N)],int)
B=numpy.array([input().split() for _ in range(N)],int)


print(numpy.dot(A,B))

#Inner and Outer

import numpy

A=numpy.array(input().split(),int)
B=numpy.array(input().split(),int)

print(numpy.inner(A,B))
print(numpy.outer(A,B))

#Polynomials

import numpy

P=list(map(float,input().split()))
x=float(input())

print(numpy.polyval(P,x))

#Linear Algebra

import numpy

N=int(input().split()[0])
A=numpy.array([input().split() for _ in range(N)],float)

print(numpy.round(numpy.linalg.det(A),2))

#Say "Hello, World!" With Python

print("Hello, World!")

#Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

    if n%2!=0 or (n>=6 and n<=20):
        print("Weird")
    else:
        print("Not Weird")

#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a+b)
    print(a-b)
    print(a*b)

#Print Function

if __name__ == '__main__':
    n = int(input())
    
    for i in range(n):
        print(i+1,end="")
    print()

#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k!=n])

#Find the Runner-Up Score! 

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
    print(max([i for i in arr if i!=max(arr)]))

#Nested Lists

if __name__ == '__main__':
    
    names=[]
    scores=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        names.append(name)
        scores.append(score)
    
    minimum=min(scores)
    scores2=[i for i in scores if i!=minimum]
    minimum=min(scores2)
    names2=[names[i] for i in range(len(names)) if scores[i]==minimum]
    names2.sort()
    for i in names2:
        print(i)

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    mean=0
    for i in student_marks[query_name]:
        mean=mean+i
    print("{:.2f}".format(mean/3,2))

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))

#Lists

if __name__ == '__main__':
    N = int(input())
    
def execute_command(list,command):
    if command[0]=='insert':
        list.insert(int(command[1]),int(command[2]))
    if command[0]=='print':
        print(list)
    if command[0]=='remove':
        list.remove(int(command[1]))
    if command[0]=='append':
        list.append(int(command[1]))
    if command[0]=='sort':
        list.sort()
    if command[0]=='pop':
        list.pop()
    if command[0]=='reverse':
        list.reverse()

list=[]
for _ in range(N):
    execute_command(list,input().split())

#sWAP cASE

def swap_case(s):
    result=''
    for c in s:
        if c.islower():
            a=c.upper()
        elif c.isupper():
            a=c.lower()
        else:
            a=c
        result=result+a
    
    return result

#String Split and Join

def split_and_join(line):
    # write your code here
    line=line.split()
    return('-'.join(line))


#What's Your Name?

def print_full_name(first, last):
    # Write your code here
    result='Hello '+first+' '+last+'! You just delved into python.'
    print(result)

#Mutations

def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]

#Find a string

def count_substring(string, sub_string):
    
    count=0
    for i in range(len(string)):
        if string[i:].find(sub_string)==0:
            count=count+1
    
    return count

#String Validators

if __name__ == '__main__':
    s = input()
    
    alphanum=False
    alpha=False
    digits=False
    lower=False
    upper=False

    for c in s:
        if c.isalnum():
            alphanum=True
        if c.isalpha():
            alpha=True
        if c.isdigit():
            digits=True
        if c.islower():
            lower=True
        if c.isupper():
            upper=True
    
    print(alphanum)
    print(alpha)
    print(digits)
    print(lower)
    print(upper)

#Text Alignment

width=int(input())

total_width=width*6

cone_width=2*width-1

#spessore (in righe) della barra orizzontale
horizontal_height=(width+1)//2
    
#altezza da fine cono a inizio barra orizzontale
tower_height=width+1

odds=[]
for i in range(width):
    odds.append(2*i+1)

for i in odds:
    string='H'*i
    print(string.center(cone_width))
    
string='H'*width+' '*(3*width)+'H'*width
for _ in range(tower_height):
    print(string.center(total_width))
    
string='H'*(width*5)
for _ in range(horizontal_height):       
    print(string.center(total_width))
    
string='H'*width+' '*(3*width)+'H'*width
for _ in range(tower_height):
    print(string.center(total_width))
        
odds.reverse()
for i in odds:
    string='H'*i
    string=string.center(cone_width)
    string=' '*(4*width)+string
    print(string)
    
#Text Wrap

def wrap(string, max_width):
    n=len(string)//max_width
    result=[]
    for i in range(n):
        result.append(string[max_width*i:max_width*(i+1)]+'\n')
    result.append(string[n*max_width:])
    return ''.join(result)

#Designer Door Mat

[N,M]=map(int,input().split())
pattern='.|.'

num=(M-3)/2

x=[]
for i in range(int((N-1)/2)):
    x.append(i)
    
for i in x:
    j=2*i+1
    string=pattern*j
    print(string.center(M,'-'))

print('WELCOME'.center(M,'-'))

x.reverse()
for i in x:
    j=2*i+1
    string=pattern*j
    print(string.center(M,'-'))

#String Formatting

def print_formatted(number):
    # your code goes here
    width=len(bin(number))-2
    for i in range(number):
        j=i+1
        string=str(j).rjust(width)+' '+oct(j)[2:].rjust(width)+' '+hex(j)[2:].rjust(width).upper()+' '+bin(j)[2:].rjust(width)
        print(string)

#Alphabet Rangoli

def print_rangoli(size):
    # your code goes here
    alphabeth=list(range(size))
    alphabeth=[chr(ord('a')+i) for i in alphabeth]
    length=4*size-3
    for i in range(size):
        j=size-i-1
        temp1=alphabeth[j:]
        temp2=temp1[1:]
        temp2.reverse()
        for k in temp1:
            temp2.append(k)
        string='-'.join(temp2)
        print(string.center(length,'-'))
    
    for i in range(size-1):
        j=i+1
        temp1=alphabeth[j:]
        temp2=temp1[1:]
        temp2.reverse()
        for k in temp1:
            temp2.append(k)
        string='-'.join(temp2)
        print(string.center(length,'-'))

#Capitalize!

def solve(s):
    l=list(s)
    temp=[]
    i=0
    while i<len(l):
        if l[i]==' ':
            j=i
            while l[i]==' ':
                i=i+1
            temp.append([j,i-j])
        else:
            i=i+1
            
    l=s.split()
    l=[i.capitalize() for i in l]
    
    string=''
    if temp[0][0]==0:
        string=' '*temp[0][1]
        temp.pop(0)
    
    for i in range(len(l)-1):
        string=string+l[i]+' '*temp[i][1]
    
    string=string+l[len(l)-1]
    
    if len(l)==len(temp):
        string=string+' '*temp[len(l)-1][1]
    
    return(string)
    
#The Minion Game

def minion_game(string):
    
    vowels=['A','E','I','O','U']
    # S e K sono i punteggi di Stuart e Kevin
    [S,K] = [0,0]
    
    for i in range(len(string)):
        if string[i] in vowels:
            K=K+len(string)-i
        else:
            S=S+len(string)-i
            
                
    if S>K:
        print('Stuart '+str(S))
    elif K>S:
        print('Kevin '+str(K))
    else:
        print('Draw')

#Merge the Tools

def merge_the_tools(string, k):
    linenum=len(string)//k
    
    t=[]
    for i in range(linenum):
        t.append(string[i*k:k*(i+1)])
        
    u=[]    
    for i in t:
        u.append(list(i))
        
    for i in range(len(u)):
        cut(u[i])
        
def cut(l):
    for i in range(len(l)-1):
        l[i+1:]=[j for j in l[i+1:] if j!=l[i]]
    l=''.join(l)
    print(l)

#Introduction to Set

def average(array):
    temp=set(array)
    count=0
    for i in temp:
        count=count+i
    return(round(count/len(temp),3))

#No Idea!

[n,m]=map(int,input().split())
array=map(int,input().split())
A=set(map(int,input().split()))
B=set(map(int,input().split()))

happiness=0
for i in array:
    if i in A:
        happiness+=1
    elif i in B:
        happiness-=1
        
print(happiness)

#Symmetric Difference

input()
N=set(map(int,input().split()))
input()
M=set(map(int,input().split()))
simmdiff=(N.difference(M)).union(M.difference(N))
simmdiff=list(simmdiff)
simmdiff.sort()
for i in simmdiff:
    print(i)

#Set .add()

N=int(input())
stamps=set()
for i in range(N):
    stamps.add(input())
print(len(stamps))

#Set .discard(), .remove() & .pop()

input()
s=set(map(int,input().split()))
commnum=int(input())

for i in range(commnum):
    commands=input().split()
    
    if len(commands)>1:
        getattr(s,commands[0])(int(commands[1]))
    else:
        getattr(s,commands[0])()

    
print(sum(s))

#Write a function

def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%4==0:
        if year%100==0:
            if year%400==0:
                leap=True
        else:
            leap=True
                
    
    return leap

#Loops

if __name__ == '__main__':
    n = int(input())
for i in range(n):
    print(i**2)

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)
