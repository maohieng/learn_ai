X1=0
X2=0
 
#X1^notX2
w1=1
w2=-1
threshold1 = -0.5
S1= X1*w1+X2*w2
def step_function_1 (S,threshold):
  if S<threshold:
    return 0
  else:
    return 1
 
#notX1^X2
w3= -1
w4= 1
S2=X1*w3+X2*w4
threshold2 = -0.5
 
y1=step_function_1(S1,threshold1)
y2=step_function_1(S2,threshold2)
print(y1)
print(y2)
 
# OR y1 v y2
 
# w5=
# w6=
# S3=
# threshold3=
w5=1
w6=1
S3=y1*w5+y2*w6
threshold3=1.5
 
def step_function_2 (S, threshold):
    if S<threshold:
        return 0
    else:
        return 1
    
y3=step_function_2(S3,threshold3)
print(y3)

