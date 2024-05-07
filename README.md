## NUMBERS

1.CODE
 ```bash
num = list(range(10))
previousnum = 0
for i in num:
    sum = previousnum + 1
    print('Current number' + str(i)+ 'previous number'+ str(previousnum)+ 'is' + str(sum))
    previousnum = i

```
