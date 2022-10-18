# 剑指offer

## 常见其他

### 斐波那契数列

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。
n<=39

n0 = 0 n1 = 1 n2 = 1 n3 = n2+n1 n4 = n2+n3 ...
    从第2项开始 每一项等于前面两项和

递归

```
class Solution {
public:
    int Fibonacci(int n) {
        //结束条件
        if( n < 2)
            return n;
         return Fibonacci(n-1) + Fibonacci(n-2);
    }
};
```

迭代

```
class Solution {
public:
    int Fibonacci(int n) {
        if(n <= 1 )
            return n;
        int n1 = 0, n2 = 1,n3 = 0;
        n -= 1;
        while(n--) {
            n3 = n1 + n2;
            n1 = n2;
            n2 = n3;
        }
        return n3;
    }
};
```

### 跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

思路：

前一个跳一级 和 前两个跳两级 都可以到达当前台阶，所以到达当前台阶的方法 是前两个之和

1 斐波那契数列

```
class Solution {
public:
    int jumpFloor(int number) {
        if( number <= 2)
            return number;
        int n1 = 1;
        int n2 = 2;
        int n3 = 0; //初始化
        number -= 2;
        while(number--) {
            n3 = n1 + n2;
            n1 = n2;
            n2 = n3;
        }
        return n3;
    }
};
```

2 递归

```
class Solution {
public:
    int jumpFloor(int number) {
        //当前台阶是前一级台阶条跳一步，前两级台阶跳两步
        if( number <= 2)
            return number;
        return jumpFloor(number-1) + jumpFloor(number-2);
    }
};
```

### 变态台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

1 暴力

```
方法一：暴力方法
设f[i] 表示 当前跳道第 i 个台阶的方法数。那么f[n]就是所求答案。

假设现在已经跳到了第 n 个台阶，那么前一步可以从哪些台阶到达呢？

如果上一步跳 1 步到达第 n 个台阶，说明上一步在第 n-1 个台阶。已知跳到第n-1个台阶的方法数为f[n-1]

如果上一步跳 2 步到达第 n 个台阶，说明上一步在第 n-2 个台阶。已知跳到第n-2个台阶的方法数为f[n-2]

。。。

如果上一步跳 n 步到达第 n 个台阶，说明上一步在第 0 个台阶。已知跳到 第0个台阶的方法数为f[0]

那么总的方法数就是所有可能的和。也就是f[n] = f[n-1] + f[n-2] + ... + f[0]

显然初始条件f[0] = f[1] = 1

所以我们就可以先求f[2]，然后f[3]...f[n-1]， 最后f[n]

class Solution {
public:
    int jumpFloorII(int number) {
        if( number < 2)
            return number;
        vector<int> f(number+1,0);
        f[0] = 1;
        f[1] = 1;
        //f[n]  = f[0]+f[1]+...+f[n-1]
        for(int i = 2; i <= number; i++)
            for(int j = 0; j < i; j++)
                f[i] += f[j];
        return f[number];
    }
};
```

2 方法二：继续优化

```
对于方法一中的：f[n] = f[n-1] + f[n-2] + ... + f[0]

那么f[n-1] 为多少呢？

f[n-1] = f[n-2] + f[n-3] + ... + f[0]

所以一合并，f[n] = 2*f[n-1]，初始条件f[0] = f[1] = 1

所以可以采用递归，记忆化递归，动态规划，递推。具体详细过程，可查看青蛙跳台阶

class Solution {
public:
    int jumpFloorII(int number) {
        if( number < 2)
            return number;
        int a = 1,b = 0;
        for(int i = 2; i <= number; i++) {
            b = a << 1;
            a = b;
        }
        return b;
    }
};
```

### 矩形覆盖

我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

比如n=3时，2*3的矩形块有3种覆盖方法：

分析：

n = 1  1 种

n = 2  2 种

n = 3  3 种

n = 4  5 种

n = 5  8 种

终止条件为 n = 1 || n = 2

其他 情况都是 f[n] = f[n-1] + f[n-2];

1 递推

```
class Solution {
public:
    int rectCover(int number) {
        if(number <= 2)
            return number;
        int a = 1;
        int b = 2;
        int c = 0; //初始化

        number -=2;
        while(number--){
            c = a + b;
            a = b;
            b = c;
        }
        return c;
    }
};
```

2 递归

```
class Solution {
public:
    int rectCover(int number) {
        if(number <= 2)
            return number;
        return rectCover(number-1) + rectCover(number-2);
    }
};
```

### 二进制中1的个数

输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。

用到了原码 反码 补码的知识

最左边是符号位，始终不变

1 二进制移位法

向右移动一位 == /2
按位与 操作 同为1 才是 1

全是整数可以让二进制数逐步/2 与 1 按位与

这里有负数，最高位保持1 ，我们可以用1逐步向右移位，再与二进制数按位与操作，解决负数问题

```
class Solution {
public:
     int  NumberOf1(int n) {
         int num = 0;
         int mark = 1;
         while(mark != 0) {
             if(mark & n)
                 num++;
             mark <<=  1;
         }
         return num;
     }
};
```

2 技巧法

对于上一种解法中，无用操作是，如果当前位是0， 还是会做判断，然后一位一位的移动。

对于上一种解法中，无用操作是，如果当前位是0， 还是会做判断，然后一位一位的移动。

整数n，进行n&(n-1)运算，会把二进制表示中最右边的1变为0

如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。

举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。

```
class Solution {
public:
     int  NumberOf1(int n) {
         int num = 0;
         while(n)
         {
             num++;
             n = n & (n-1);
         }
         return num;
     }
};
```

### 数值的 整数 次方

给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

保证base和exponent不同时为0

1  暴力

遇到指数为负数 x的-2次方  == 1/x^2

```
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent < 0)
        {
            base = 1/base;
            exponent = - exponent;
        }
        double sum = 1.0;
        //暴力遍历
        while(exponent > 0)
        {
            sum *= base;
            exponent--;
        }
        return sum;
    }
};
```

2 非递归 

已知6可以表示成二进制110
6 = 0 * 2^0 + 1 * 2^1 + 1 * 2^2
对于二进制数，遇到位数是1的就乘到答案中。

方法相当于遍历n的二进制位，是1就乘进结果
时间复杂度：O(logn)，因为n的二进制位个数为logn
空间复杂度：O(1)

```
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent < 0)
        {
            base = 1/base;
            exponent = - exponent;
        }
        double x = base;
        double sum = 1.0;
        while(exponent)
        {
            if(exponent & 1)
            {
                sum *= x;
            }
            //向左移动一位 指数+1
            x *= x;
            exponent >>= 1;
        }
        return sum
    }
};
```

3 递归

假设我们求 x^8 ，x^8 = (x^4)^2
即x^n = x^(n/2)^2;

当n = 奇数

即x^n = x^(n/2)^2 * x;

```
class Solution {
public:
    double Power(double base, int exponent) {
        //x^4 = (x^2)^2
        if(exponent == 0)
            return 1.0;
        if(exponent < 0)
        {
            base = 1/base;
            exponent = -exponent;
        }
        //递归
        double ret = Power(base, exponent/2);
        //奇数次幂需要额外处理
        if(exponent & 1)
            ret = ret * ret * base;
        else
            ret = ret * ret;
        return ret;
    }
};
```

### 求 1- n的整数中 1出现的次数

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

暴力

```
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int ret = 0;
        for(int i = n; i >= 1; i--)
            for(int j = i; j >= 1; j/=10)
            {
                if(j%10 == 1)
                    ret++;
            }
        return ret;
    }
};
```

### 丑数

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

丑数能够分解成2^x3^y5^z,
所以只需要把得到的丑数不断地乘以2、3、5之后并放入他们应该放置的位置即可，

1乘以 （2、3、5）=2、3、5；2乘以（2、3、5）=4、6、10；3乘以（2、3、5）=6,9,15；5乘以（2、3、5）=10、15、25；

```
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index <= 0)
            return 0;
        //三个指针指向三个潜在成为最小丑数的位置
        int p2 = 0, p3 = 0, p5 =0;
        vector<int> result(index,0);
        result[0] = 1;
        for(int i = 1; i < index; i++)
        {
            //取最小值作为新的最小 丑数
            result[i] = min(result[p2]*2,min(result[p3]*3,result[p5]*5));
            //去除重复
            if(result[i] == result[p2]*2)
                p2++;
            if(result[i] == result[p3]*3)
                p3++;
            if(result[i] == result[p5]*5)
                p5++;
        }
        return result[index-1];
    }
};
```

### 求1+2+3+...+n

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）

2 数学

sum = ((1+n)*n)/2
```
 int m = (int)Math.pow(n, 2)+n;
       return m>>1;
```

3 递归
递归函数f(n)表示求1-n的和。
递推公式：f(n) = f(n-1) + n
递归终止条件：f(1) = 1

```
class Solution {
public:
    int Sum_Solution(int n) {
       //f[n] = 1 + ... + n
       //f[n] = f[n-1] + n
       //递归终止条件
        if(n == 1)
            return n;
        return Sum_Solution(n-1) + n;
    }
};
```

### 位运算实现加法

```
class Solution {
public:
    int Add(int num1, int num2)
    {
        //两个数异或就是不进位的加法
        int sum1 = num1 ^ num2;
        //两个数按位与是取进位
        int sum2 = (num1 & num2) << 1;
        
        return sum1 +  sum2;
    }
};
```

### 滑动窗孔的最大值

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
窗口大于数组长度的时候，返回空

1 暴力

```
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        if(num.empty() || num.size() < size|| size <= 0)
            return {};
        vector<int> ans;
        int maxNum = INT_MIN;

      for(int i = 0; i <= num.size() - size; i++){
            int maxNum = num[i];
            for(int j = i; j < i + size; j ++){
                maxNum = max(maxNum, num[j]);
            }
            ans.push_back(maxNum);
        }
        return ans;
    }
};
```

队列

辅助队列的方式

遍历nums

1 如果队列非空，则将队列中比当前要插入结点小的元素出队
2 新的元素入队
3 检查队首元素是否还在窗口中(que.front()+size < index)
检查窗口已满，满了就将队首元素作为本次的窗口最大值存储到结果集

```
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        if(num.empty() || num.size() < size || size <= 0)
            return {};
        
        vector<int> ret;
        //辅助队列，存元素的下标 （用于判断是否在窗口内）
        deque<int> que;
        int len = num.size();
        for(int i = 0; i < len; i++)
        {
            //如果队列非空，则将队列中比当前要插入结点小的元素出队
            while(!que.empty() && num[que.back()] < num[i])
            {
                que.pop_back();
            }
            //新的元素的下标入队
            que.push_back(i);
            //检查队首元素是否在窗口中
            if(!que.empty() && que.front() + size <= i)
            {
                que.pop_front();
            }
            //判断窗口是否已满
            if(i >= size-1)
            {
                ret.push_back(num[que.front()]);
            }
        }
         return ret;
    }
};
```

## 数组题

## 二维数组中的查找

题目描述：

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

![二维数组的查找](photos/二维数组的查找.png)

* 解法1

```
算法流程：
从矩阵 matrix 左下角元素（索引设为 (i, j) ）开始遍历，并与目标值对比：
偏大
    当 matrix[i][j] > target 时，执行 i-- ，即消去第 i 行元素；
偏小
    当 matrix[i][j] < target 时，执行 j++ ，即消去第 j 列元素；
目标
    当 matrix[i][j] = target 时，返回 truetrue ，代表找到目标值。
索引越界则找不到
    若行索引或列索引越界，则代表矩阵中无目标值，返回 falsefalse 。

class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        if(array.empty() || array[0].empty())
            return false;
        int row = array.size()-1;
        int col = array[0].size()-1;
        //左下角为参考点
        int i = row;
        int j = 0;
        
        while(j <= col && i >= 0)
        {
            //偏大
            if(array[i][j] > target)
                i--;
            //偏小
            else if(array[i][j] < target)
                j++;
            else
                return true;
        }
        //索引越界了还没找到
        return false;
    }
};

复杂度分析：
时间复杂度 O(M+N)O(M+N) ：其中，NN 和 MM 分别为矩阵行数和列数，此算法最多循环 M+NM+N 次。
空间复杂度 O(1)O(1) : i, j 指针使用常数大小额外空间。
```

* 右上角为参考点

1）我设置初始值为右上角元素，arr[0][5] = val，目标tar = arr[3][1]
2）接下来进行二分操作：
3）如果val == target,直接返回
4）如果 tar > val, 说明target在更大的位置，val左边的元素显然都是 < val，间接 < tar，说明第 0 行都是无效的，所以val下移到arr[1][5]
5）如果 tar < val, 说明target在更小的位置，val下边的元素显然都是 > val，间接 > tar，说明第 5 列都是无效的，所以val左移到arr[0][4]
6）继续步骤2）

复杂度分析
时间复杂度：O(m+n) ，其中m为行数，n为列数，最坏情况下，需要遍历m+n次。
空间复杂度：O(1)


```
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
       if(array.empty() || array[0].empty())
           return false;
        int row = array.size()-1;
        int col = array[0].size()-1;
        int i = 0;
        int j = col;
        while(i <= row && j >= 0)
        {
            if(target == array[i][j])
                return true;
            else if(target > array[i][j])
                i++;
            else
                j--;
        }
        return false;
    }
};
```

### 旋转数组的最小数字

题目描述

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

示例1
[3,4,5,1,2]
返回值
1

方法1 暴力

```
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.empty())
            return 0;
        int min = INT_MAX;
        for(int v : rotateArray)
            if( v < min)
                min = v;
        return min;
    }
};
```

方法2 二分查找

没有给出target的二分查找，难就难在arr[mid]跟谁比

我们的目的是：当进行一次比较时，一定能够确定答案在mid的某一侧。一次比较为 arr[mid]跟谁比的问题。

一般的比较原则有：

1 如果有目标值target，那么直接让arr[mid] 和 target 比较即可。
2 如果没有目标值，一般可以考虑端点
这里我们把target看作是右端点，来进行分析，那就要分析以下三种情况，看是否可以达到上述的目标。

情况1，arr[mid] > target：4 5 6 1 2 3
arr[mid] 为 6， target为右端点 3， arr[mid] > target, 说明[first ... mid] 都是 >= target 的，因为原始数组是非递减，所以可以确定答案为 [mid+1...last]区间,所以 first = mid + 1

情况2，arr[mid] < target:5 6 1 2 3 4
arr[mid] 为 1， target为右端点 4， arr[mid] < target, 说明答案肯定不在[mid+1...last]，但是arr[mid] 有可能是答案,所以答案在[first, mid]区间，所以last = mid;

情况3，arr[mid] == target:
如果是 1 0 1 1 1， arr[mid] = target = 1, 显然答案在左边
如果是 1 1 1 0 1, arr[mid] = target = 1, 显然答案在右边

所以这种情况，不能确定答案在左边还是右边，那么就让last = last - 1(或者start++);慢慢缩少区间，同时也不会错过答案。

```
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.empty())
            return 0;
        int begin = 0;
        int end = rotateArray.size()-1;
        int mid = 0;
        //进行二分查找，以end为target
        while(begin < end )
        {
            mid = (begin+end)/2;
            //元素数组是非递减，所以结果在[mid+1,end]
            if(rotateArray[mid] > rotateArray[end])
            {
                begin = mid+1;
            }
            //中间值小于尾数，最下值在[begin,mid]
            else if(rotateArray[mid] < rotateArray[end])
            {
                end = mid;
            }
            //相等的时候可能有特殊情况如 11101，缩小边界即可
            else
            {
                end--;
            }
        }
        return rotateArray[end];
    }
};
```

### 调整数组顺序，使得奇数位于偶数前面

题目描述

入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

1. in-place算法
如果不开辟额外数组该怎么做呢？
初始化操作：记录一个变量i表示已经将奇数放好的下一个位置，显然最开始i=0,表示还没有一个奇数放好。
j 表示数组的下标，初始值为0， 表示从下标0开始遍历。

如果遇到偶数，j++
如果遇到奇数,假设位置为j，就将此奇数插入到i所指的位置，然后i往后移动一个位置，在插入之前，显然会涉及到数据的移动，也就是将[i,j-1]整体往后移动。
直到整个数组遍历完毕，结束

```
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        //in-place算法，遇到一个就后移
        int size = array.size();
        if(size < 2)
            return;
        int index = 0; //用于记录上一个奇数的位置的下一个位置
        for(int i = 0; i < size; i++)
        {
            //遇到奇数就
            if(array[i] & 1)
            {
                int temp = array[i];
                //上个奇数的位置开始到i-1都要向后移动一个位置
                for(int j = i-1; j >= index ; j--)
                {
                    array[j+1] = array[j];
                }
                //腾出位置后插入奇数，index前移
                array[index++] = temp;
            }
        }
    }
};
```

2 辅助数组

```
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        if(array.size() < 2)
            return ;
        vector<int> vec1;
        vector<int> vec2;
        for(int v : array) {
            if(v & 1)
                vec1.push_back(v);
            else
                vec2.push_back(v);
        }
        int i;
        for(i = 0; i < vec1.size(); i++) {
            array[i] = vec1[i];
        }
        for(int j = 0; j <vec2.size(); j++)
            array[i+j] = vec2[j];
    }
};
```

### 把数组的所有奇数移动到前面，保持奇数与奇数，偶数与偶数的相对位置

1. in-place算法
如果不开辟额外数组该怎么做呢？
初始化操作：记录一个变量i表示已经将奇数放好的下一个位置，显然最开始i=0,表示还没有一个奇数放好。
j 表示数组的下标，初始值为0， 表示从下标0开始遍历。

如果遇到偶数，j++
如果遇到奇数,假设位置为j，就将此奇数插入到i所指的位置，然后i往后移动一个位置，在插入之前，显然会涉及到数据的移动，也就是将[i,j-1]整体往后移动。
直到整个数组遍历完毕，结束

```
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int size = array.size();
        if(size < 2)
            return;
        int i = 0; //记录已经放好的奇数的下一个位置
        for(int j = 0; j < size; j++)
        {
            //奇数
            if(array[j] & 1)
            {
                int tmp = array[j];
                //向后移动一位，腾出位置
                for(int k  = j-1; k >= i; --k)
                {
                    array[k+1] = array[k];
                }
                array[i++] =  tmp;
            }
        }
    }
};
```

2 辅助数组

```
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        if(array.size() < 2)
            return ;
        vector<int> vec1;
        vector<int> vec2;
        for(int v : array) {
            if(v & 1)
                vec1.push_back(v);
            else
                vec2.push_back(v);
        }
        int i;
        for(i = 0; i < vec1.size(); i++) {
            array[i] = vec1[i];
        }
        for(int j = 0; j <vec2.size(); j++)
            array[i+j] = vec2[j];
    }
};
```

### 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

思路

题目抽象：给一个二维矩阵，顺时针转圈打印矩阵。
转圈是说：把矩阵最外层看成一个圈
方法一：转圈打印
如果有个方法是顺时针转圈打印矩阵，那么我们可以先打印最外圈，然后再打印次外圈。
如图：

最外层为 [1 2 3 4 8 12 16 15 14 13 9 5]
次外层为 [6 7 11 10]
这里只有2层。
我们可以用矩阵的左上角和右下角唯一表示一个矩阵。设左上角坐标为(lx,ly), 右下角坐标为(rx,ry)

第一步：打印 [1 2 3 4]

第二步：打印 [8 12 16]

第三步：打印 [15 14 13]

第四步：打印 [8 5]
因此可实现函数的代码为：

// matrix原二维vector，ret 存打印结果的vector
void print(int lx, int ly, int rx, int ry, vector<vector<int>> &matrix, vector<int> &ret) {

    for (int j=ly; j<=ry; ++j) ret.push_back(matrix[lx][j]);
     for (int i=lx+1; i<=rx; ++i) ret.push_back(matrix[i][ry]);
     int h = rx - lx + 1;
     if (h > 1) // 只有一行，不需要第三步
         for (int rj=ry-1; rj>=ly; --rj) ret.push_back(matrix[rx][rj]);
     int w = ry - ly + 1;
     if (w > 1) // 只有一列不需要第四步
         for (int ri = rx-1; ri>=lx+1; --ri) ret.push_back(matrix[ri][ly]);
 }

转圈打印函数已经实现好了，那么接下来每次打印一圈就好。缩小一圈就是lx+1, ly+1, rx-1, ry-1

```
class Solution {
public:
    void print(vector<vector<int> > matrix,int lx,int ly, int rx, int ry, vector<int> &ret)
    {
        //上边从左到右
        for(int j = ly; j <=  ry; j++)
            ret.push_back(matrix[lx][j]);
        //右边从上到下
        for(int i = lx+1; i <= rx; i++)
            ret.push_back(matrix[i][ry]);
         //如果只有一行，不用第三步
        if((rx - lx) > 0) {
            //下边从右到左
            for(int j = ry-1; j >= ly; j--)
                ret.push_back(matrix[rx][j]);
        }
          //如果只有一列，不用第四步
        if((ry-ly) > 0) {
            //左边从下到上
            for(int i = rx-1; i >= lx+1; i--)
                ret.push_back(matrix[i][ly]);
        }
    }
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int row = matrix.size();
        if(row == 0)
            return{};
        int col = matrix[0].size();
        if(col == 0)
            return {};
        //以左上角(lx,ly)和右下角(rx,ry)唯一表示一个矩阵
        int lx = 0, ly = 0;
        int rx = row-1, ry = col-1;
        vector<int> ret;
        while(lx <= rx && ly <= ry)
            print(matrix,lx++,ly++,rx--,ry--,ret);
        return ret;
    }
};
```

### 数组中出现超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0

1 哈希

```
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.empty())
            return 0;
        //哈希
        map<int,int> mp;
        //建立哈希表
        for(int v : numbers)
        {
            ++mp[v];
        }
        //遍历哈希表
        for(int v : numbers)
        {
            if(mp[v] > numbers.size()/2)
                return v;
        }
        return 0;
    }
};
```

2 排序法

出现次数过半的数字排序后 应该出现在中间，然后判断一下

```
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.empty())
            return 0;
        sort(numbers.begin(),numbers.end());
        int target = numbers[numbers.size()/2];
        int count = 0;
        for(int i = 0; i < numbers.size(); i++)
        {
            if(target  == numbers[i])
                count++;
            if(target < numbers[i])
                break;
        }
        if(count > numbers.size()/2)
            return target;
        return 0;
    }
};
```

### 最小k个数

* 题目描述
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

* 直接排序法

```
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if( input.size() < k)
            return {};
        sort(input.begin(), input.end());
        vector<int> ret;
        for(int i = 0; i < k; i++)
            ret.push_back(input[i]);
        return ret;
    }
};
```

* 堆排序

建立一个容量为k的大根堆的优先队列。遍历一遍元素，如果队列大小<k,就直接入队，否则，让当前元素与队顶元素相比，如果队顶元素大，则出队，将当前元素入队

```
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(input.size() < k || k <= 0)
            return {};
        vector<int> ret;
        //维护一个大根堆
        priority_queue<int,vector<int> > que;
        for( int val : input)
        {
            //不足k个则直接加入
            if(que.size() < k)
                que.push(val);
            //与堆顶的 最大值比较
            else
            {
               if(val < que.top())
               {
                   que.pop();
                   que.push(val);
               }
            }
        }
        //将堆的元素都添加到ret集合中
        while(!que.empty())
        {
           ret.push_back(que.top());
           que.pop();
        }
        return ret;
    }
};
```

3 快排

对数组[l, r]一次快排partition过程可得到，[l, p), p, [p+1, r)三个区间,[l,p)为小于等于p的值
[p+1,r)为大于等于p的值。
然后再判断p，利用二分法

如果[l,p), p，也就是p+1个元素（因为下标从0开始），如果p+1 == k, 找到答案
2。 如果p+1 < k, 说明答案在[p+1, r)区间内，
3， 如果p+1 > k , 说明答案在[l, p)内

```
class Solution {
public:
    int quickSort(vector<int> &input,int left, int right)
    {
        //暂存枢纽值
        int pivot = input[right];
        while(true)
        {
            //从左边找一个比枢纽值大的移动到右边
            while(left < right && input[left] <= pivot)
                left++;
            if(left >= right)
                break;
            //移动
            input[right] = input[left];
            right--;
            
            //从右边找一个比枢纽值小的移动到左边
            while(left < right && input[right] >= pivot)
                right--;
            if( left >= right)
                break;
            input[left] = input[right];
            left++;
        }
        
        //移动枢纽值
        input[left] = pivot;
        return left;
    }
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(input.size() < k || k <= 0)
            return {};
        int left = 0;
        int right = input.size();
        
        while(left < right)
        {
            //进行一次快排，获取枢纽值
            int  pivot = quickSort(input,left,right-1);
            if( pivot+1 == k)
                return vector<int> {input.begin(),input.begin()+k};
            //k值在右边
            else if(pivot+1 < k)
            {
                left = pivot + 1;
            }
            //k值在左边
            else
            {
                right = pivot;
            }
        }
        return {};
    }
};
```

快排以左边一个为枢纽值

```
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k <= 0 || k > input.size())
            return {};
        int left = 0, right = input.size();
        while(left < right)
        {
            int p = quickSort(input,left,right-1);
            //找到第k大的数字的位置
            if( p+1 == k)
            {
                return vector<int> ({input.begin(),input.begin()+k});
            }
            //第K大数在枢纽值右边
            else if( k > p+1)
            {
                left = p+1;
            }
            //第K大数在枢纽值左边
            else
                right = p;
        }
        return input;
    }
    //快排一次，返回枢纽值的下标
    int quickSort(vector<int> &input, int left, int right)
    {
        int pivot = input[left];
        while(true)
        {
            //右边找一个小于枢纽值的元素移动到左边
            while(left < right && input[right] >= pivot)
                right--;
            //找不到则退出
            if(left >= right)
                break;
            //移动
             input[left] = input[right];
            left++;
            //左边找一个大于枢纽值得元素移动到右边
            while(left < right && input[left] <= pivot)
                left++;
            //找不到则退出
            if(left >= right)
                break;
            //移动
            input[right] = input[left];
            right--;
        }
        //left == right 移动枢纽值
        input[left] = pivot;
        return left;
    }
};

```

### 连续子数组的最大和

给一个数组，返回它的最大连续子序列的和

方法一：动态规划
状态定义：dp[i]表示以i结尾的连续子数组的最大和。所以最终要求dp[n-1]
状态转移方程：dp[i] = max(array[i], dp[i-1]+array[i])
解释：如果当前元素为整数，并且dp[i-1]为负数，那么当然结果就是只选当前元素
技巧：这里为了统一代码的书写，定义dp[i]表示前i个元素的连续子数组的最大和，结尾元素为array[i-1]

```
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int size = array.size();
        vector<int> dp(size+1,1);

        dp[0] = 0; //表示第一个不用，没有元素

        int ret = array[0]; //维持最大值

        for(int i = 1; i <= size; i++)
        {
            dp[i] = max(array[i-1],dp[i-1] + array[i-1]);
            if(dp[i] > ret)
                ret = dp[i];
        }
        return ret;
    }
};
```

2 dp[0]也使用

```
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(array.empty())
            return 0;
        //动态规划：当前元素的值为：前一元素的值 与 前一元素+数组当前位置值
        int size = array.size();
        vector<int> dp(size,1);
        
        dp[0] = array[0]; 
        int ret = array[0];
        
        for(int i = 1; i < size; i++)
        {
            dp[i] = max(array[i],dp[i-1] + array[i]);
            //更新
            if(dp[i] > ret)
                ret = dp[i];
        }
        
        return ret;
    }
};
```

### 把数组排成最小数

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

1 全排序列

```
class Solution {
public:
    void perm(int position,vector<int> &numbers,string &ret)
    {
        //递归结束条件
        if(position + 1 == numbers.size())
        {
            string tmp = "";
            for(int val : numbers)
            {
                tmp += to_string(val);
            }
            ret = min(ret,tmp);
            return;
        }
        //遍历递归
        for(int i = position; i < numbers.size(); i++)
        {
            //先交换
            swap(numbers[position], numbers[i]);
            //递归找到满足条件
            perm(position+1,numbers,ret);
            //回溯
            swap(numbers[position],numbers[i]);
        }
    }
    string PrintMinNumber(vector<int> numbers) {
        //初始化为最大值 
        string ret(numbers.size(),'9');
        perm(0,numbers,ret);
        return ret;
    }
};
```

 2 自主排序：
    先排序再拼接

排序方法

仿函数
struct Com {
    bool operator() (string a, string b) {
     return a + b < b + a;
    }
};
sort(str.begin(), str.end(), Com()); // Com()为临时对象

lambda表达式
// 1. 匿名lambda表达式
sort(str.begin(), str.end(), [](string a, string b) {
     return a + b < b + a;
});
// 2.具名lambda表达式
auto lam = [](string a, string b) {
     return a + b < b + a;
 };
sort(str.begin(), str.end(), lam);;

函数指针
bool static com(string a, string b) {
    return a + b < b + a;
}
//加static的原因：类成员函数有隐藏的this指针,static 可以去this指针
sort(str.begin(), str.end(), com);

```
class Solution {
public:
    static bool compare(const string &str1, const string &str2)
    {
        string  s1 = str1 + str2;
        string s2 = str2 + str1;
        return s1 < s2;
    }
    string PrintMinNumber(vector<int> numbers) {
        int size = numbers.size();
        string result;
        vector<string> vec;
        if(size == 0)
            return result;
        //先将所有的数字排序放到vec,再拼接
        for(int i = 0; i < size; i++)
        {
            vec.push_back(to_string(numbers[i]));
        }
        sort(vec.begin(),vec.end(),compare);
        for(int i = 0; i < vec.size(); i++)
            result += vec[i];
        return result;
    }
};
```

### 数组中的逆序对

题目描述
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

1 暴力

```
class Solution {
public:
    const int mod = 1000000007;
    int InversePairs(vector<int> data) {
        int count = 0;
        int size = data.size();
        for(int i = 0; i < size; i++)
        {
            for(int j = i+1; j < size; j++)
            {
                if(data[i] > data[j])
                {
                    ++count;
                    count %= mod;
                }
            }
        }
        return count;
    }
};
```

2 归并排序

归并排序的过程，主要有以下两个操作：
    递归划分整个区间为基本相等的左右两个区间
    合并两个有序区间

如果区间有序会有什么好处吗？当然，如果区间有序，比如[3,4] 和 [1,2]
如果3 > 1, 显然3后面的所有数都是大于1， 这里为 4 > 1, 明白其中的奥秘了吧。所以我们可以在合并的时候利用这个规则。

```
class Solution {
public:
    const int mod = 1000000007;
    //合并函数
    void merge(vector<int> &arr,vector<int> &vec,int left,int mid,int right,int &ret)
    {
        int i = left;
        int j = mid+1;
        int k = 0;
        
        //
        while( i <= mid && j <= right)
        {
            //满足逆序对的条件
            if(arr[i] > arr[j])
            {
                vec[k++] = arr[j++];
                //有序序列 3 > 1 则 3后面都 > 1
                ret += (mid - i +1);
                ret %= mod;
            }
            else
            {
                vec[k++] = arr[i++];
            }
        }
            //处理剩余的部分
            while( i <= mid)
            {
                vec[k++] = arr[i++];
            }
            while(j <= right)
            {
                vec[k++] = arr[j++];
            }
            //排序好回复到原数组
            for(k = 0, i = left; i <= right; ++i,++k)
            {
              arr[i] = vec[k];
            }
    }
    //递归划函数
    void merge_sort(vector<int> &arr,vector<int> &vec,int left,int right,int &ret)
    {
        //递归结束条件，只剩一个元素
        if(left >= right)
            return;
        int mid = left + (right-left)/2;
        //递归
        merge_sort(arr, vec, left, mid, ret);
        merge_sort(arr, vec, mid+1, right, ret);
        //合并
        merge(arr,vec,left,mid,right,ret);
    }
    
    //调用函数
    int InversePairs(vector<int> data) {
        int ret = 0;
        //额外的容器存储归并的结果
        vector<int> vec(data.size());
        merge_sort(data,vec,0,data.size()-1,ret);
        return ret;
    }
};
```

### 数字在升序数组中出现的次数

统计一个数字在升序数组中出现的次数。

1 直接暴力

class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        //暴力
        int count = 0;
        for(int v : data)
        {
            if(v == k)
                ++count;
        }
        return count;
    }
};

二分

下界定义为：如果存在目标值，则指向第一个目标值，否则，如果不存在， 则指向大于目标值的第一个值。
上界定义为：不管目标值存在与否，都指向大于目标值的第一个值。
如下图所示：

```
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        if(data.empty())
            return 0;
        int left = 0, right = data.size();
        int mid = 0;
        //寻找左边界，数第一次出现的位置
        while(left < right)
        {
            mid  = left + (right-left+)/2;
            if(data[mid] == k)
            {
                break;
            }
                left = mid +1;
            else
               right = mid;
        }
        int lbound = left;
        //寻找右边界，数最后一次出现的位置
        left = 0; right = data.size();
        while(left < right)
        {
            mid = left + (right-left)/2;
            if(data[mid] <= k)
                left = mid+1;
            else
                right = mid;
        }
        int rbound = left;
        return rbound - lbound;
    }
};
```

3 二分的思想，找到该数字出现的中间位置，然后向左右找

```
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        if(data.empty())
           return 0;
        //二分思想找该数字
        int left = 0;
        int right = data.size()-1;
        int mid = 0;
        int count = 0;
        //找左边界
        while(left < right)
        {
            mid =  (left+right)/2;
            //找到出现的中间位置
            if(data[mid] == k)
            {
                break;
            }
            //目标在左边
            else if(data[mid] < k)
            {
                left = mid+1;;
            }
            //目标
            else
            {
                right = mid;
            }
        }
        int index = mid;
        //向左边找
        while(index >= 0 && data[index] == k)
        {
            index--;
            count++;
        }
        //向右找
        index = mid+1;
        while(index <= data.size()-1 && data[index] == k)
        {
            index++;
            count++;
        }
        return count;
    }
};
```

### 数组中只出现一次的数字

题目描述
一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

1 哈希

```
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        if(data.empty())
            return;
        map<int,int> mp;
        for(int v : data)
            ++mp[v];
        vector<int> ret;
        for(const int v :data)
        {
            if(ret.size() == 2)
                break;
            if(mp[v] == 1)
              ret.push_back(v);
        }
        *num1 = ret[0];
        *num2 = ret[1];
    }
};
```

2 位运算

如果a、b两个值不相同，则异或结果为1。如果a、b两个值相同，异或结果为0。

```
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        if(data.empty())
            return;
        int ret = 0;
        //所有数与ret异或，得到的是只出现一次的两个数的异或结果
        for(int k : data)
            ret = ret ^ k;
        //两个不同的数，一定有一个二级制为位是不同的
        ret &= -ret;
        *num1 = 0;
        *num2 = 0;
        //遍历，将数分为两个部分，最后的结果就是两个出现一次的数
        for(int k : data)
        {
            if(k & ret)
                *num1 ^= k;
            else
                *num2 ^= k;
        }
    }
};
```

### 和为S的连续正数序列

输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序

1 暴力

```
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> ret;
        //左边界
        for(int i = 1;  i <= sum/2; i++)
        {
            //右边界
            for(int j = i+1; j < sum; j++)
            {
                //求边界内的区间和
                int temp = 0;
                for(int k = i; k <= j; k++)
                    temp += k;
                //满足结果
                if(sum == temp)
                {
                    vector<int> res;
                    for(int k = i; k <= j; k++)
                    {
                        res.push_back(k);
                    }
                    //加入结果集
                    ret.push_back(res);
                }
                //后续的都会大于sum
                else if(temp > sum)
                    break;
            }
        }
        return ret;
    }
};
```

2 前缀和
sum[i]表示前i个数的和。比如sum[1] = 1,表示前一个数的和为1，sum[2] = 3, 表示前2个数的和为3.现在我们要求区间[2,4]表示求第2,3,4个数的和，就等于sum[4] - sum[1] = 9

```
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int> > ret;
        int temp = 0;
        for(int i = 1; i <= sum/2; i++)
            for(int j = i+1; j < sum; j++)
            {
                //前缀累加
                temp += j;
                if(sum == temp)
                {
                   vector<int> ans;
                   for (int k=i; k<=j; ++k)
                       ans.push_back(k);
                   ret.push_back(ans);
                }
                else if (temp > sum)
                {
                    // 提前剪枝
                    temp = 0;
                    break;
                }
            }
        return ret;
    }
};
```

3 窗口

扩大窗口，j += 1
缩小窗口，i += 1
算法步骤：
初始化，i=1,j=1, 表示窗口大小为0
如果窗口中值的和小于目标值sum， 表示需要扩大窗口，j += 1
否则，如果狂口值和大于目标值sum，表示需要缩小窗口，i += 1
否则，等于目标值，存结果，缩小窗口，继续进行步骤2,3,4

```
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int> > ret;
        //窗口的左右边界
        int left = 1;
        int right = 1;
        //窗口内的区间和
        int temp = 0;
        while( left <= sum/2)
        {
            //区间和偏小，右边界前移
            if(temp < sum)
            {
                temp += right;
                right++;
            }
            //区间和偏大,左边界右移动
            else if(temp > sum)
            {
                temp -= left;
                left++;
            }
            //满足条件的结果集
            else
            {
                vector<int> res;
                for(int k = left; k < right; k++)
                {
                    res.push_back(k);
                }
                ret.push_back(res);
                //左边界前移
                temp -= left;
                left++;
            }
        }
        return ret;
    }
};
```

### 和为S的两个数字

题目描述
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

1 哈希

```
#include<unordered_map>
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        if(array.empty())
            return {};
        //结果集合
        pair<int,int> ret;
        int minSize = INT_MAX;
        //哈希表 {值：下标 }
        unordered_map<int,int> mp;
        
        //先逐个加入哈希表
        for(int i = 0; i < array.size(); i++)
            mp[array[i]] = i;
        
        for(int i = 0; i < array.size(); i++)
        {
            
            if(mp.find(sum-array[i]) != mp.end())
            {
                int j = mp[sum-array[i]];
                //满足条件且不重复
                if(j > i && array[i] * array[j] < minSize)
                {
                    minSize = array[i] *  array[j];
                    ret = {i,j};
                }
            }
        }
        if(ret.first == ret.second)
            return {};
        else
            return vector<int> ({array[ret.first],array[ret.second]});
    }
};
```

2 双指针

如下：
1.初始化：指针i指向数组首， 指针j指向数组尾部
2. 如果arr[i] + arr[j] == sum , 说明是可能解
3. 否则如果arr[i] + arr[j] > sum, 说明和太大，所以--j
4. 否则如果arr[i] + arr[j] < sum, 说明和太小，所以++i

```
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        if(array.empty())
            return {};
        int minSize = INT_MAX;
        pair<int,int> ret;
        int left = 0; int right = array.size()-1;
        while(left < right)
        {
            int temp = array[left] + array[right];
            //满足条件
            if(temp == sum)
            {
                //更新最小乘积
                if(temp < minSize)
                {
                    minSize = temp;
                    ret = {left,right};
                }
                //两边向中间 移动
                left++;
                right--;
            }
            //偏小
            else if(temp < sum)
            {
                left++;
            }
            //偏大
            else
            {
                right--;
            }
        }
        if(ret.first == ret.second)
            return {};
        return vector<int> ({array[ret.first],array[ret.second]});
    }
};
```

## 链表题

### 从尾到头打印链表

输入一个链表，按链表从尾到头的顺序返回一个ArrayList

* 解法1：链表的指针操作
  * 先反转链表再顺序遍历

```
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        if(head == nullptr)
            return {};
        //反转链表
        ListNode *pre = nullptr;
        ListNode *cur = head;
        ListNode *next = nullptr;
        //遍历反转
        while(cur)
        {
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        ListNode *newHead = pre;
        //遍历新链表，生成结果集
        vector<int> ret;
        while(newHead)
        {
            ret.push_back(newHead->val);
            newHead = newHead->next;
        }
        return ret;
    }
};

解法2：递归遍历，逐个加入结果集合

先递归到最后一个结点，然后逐步返回的时候按倒序填加到结果集

class Solution {
public:
    void dfs(ListNode *node, vector<int> &ret)
    {
        //递归结束条件-到了链表尾
        if(node == nullptr)
            return;
        dfs(node->next,ret);
        ret.push_back(node->val);
    }
    vector<int> printListFromTailToHead(ListNode* head) {
        if(head == nullptr)
            return {};
        vector<int> ret;
        dfs(head,ret);
        return ret;
    }
};
```

### 链表的倒数第K个节点

题目描述
输入一个链表，输出该链表中倒数第k个结点。

方法1：快慢指针
    快指针先走k步
    接着快慢指针一起走
    快指针到了结尾，慢指针指向的就是目标

```
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        //快指针先K步，然后快慢指针一起走，快指针到了尾，慢指针到了倒数第k个
        if(pListHead == nullptr)
            return pListHead;
        ListNode *fast = pListHead;
        ListNode *slow = pListHead;
        for(int i = 0; i < k; i++)
        {
            //防止k大过链表长度的问题
            if(fast != nullptr)
                fast = fast->next;
            else
                return nullptr;
        }
        //快慢指针同时移动
        while(fast)
        {
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};
```

### 反转链表

1 非递归

```
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if( pHead == nullptr)
            return pHead;
        ListNode *prev = nullptr;
        ListNode *cur = pHead;
        ListNode *next = nullptr;

        while(cur)
        {
            next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }
};
```

2 递归

```
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if(pHead == nullptr || pHead->next == nullptr)
            return pHead;
        //递归
        ListNode *p = ReverseList(pHead->next);
        
        pHead->next->next = pHead;
        pHead->next = nullptr;
        return p;
    }
};
```

### 合并两个有序链表

题目描述
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

迭代

```
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == nullptr)
            return pHead2;
        if(pHead2 == nullptr)
            return pHead1;
        
        //辅助头节点
        ListNode *head = new ListNode(-1);
        ListNode *p = head;
        //每次取出最小的
        while(pHead1 && pHead2)
        {
            if(pHead1->val < pHead2->val)
            {
                p->next = pHead1;
                pHead1 = pHead1->next;
            }
            else
            {
                p->next = pHead2;
                pHead2 = pHead2->next;
            }
            p = p->next;
        }
        //处理两者剩余的非空链表
        p->next =  pHead1 == nullptr ? pHead2 : pHead1;
        return head->next;
    }
};
```

递归

```
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(!pHead1)
            return pHead2;
        else if(!pHead2)
            return pHead1;

        if(pHead1->val < pHead2->val)
        {
            pHead1->next = Merge(pHead1->next,pHead2);
            return pHead1;
        }
        else
        {
            pHead2->next = Merge(pHead2->next, pHead1);
            return pHead2;
        }

    }
};
```

* 递归

1 功能：反转链表的过程中，从后往前将结果加入vector返回
2 递归结束条件：cur == nullptr cur->next == nullptr
3 从前往后缩小距离

```
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    ListNode * reverseList(ListNode *head)
    {
        //结束条件-最后一个节点
        if(head->next == nullptr)
           return head;
        ListNode *p = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return p;
    }
    vector<int> printListFromTailToHead(ListNode* head) {
        //递归的的功能是获取反转链表的新首元节点
        ListNode *p = reverseList(head);
        vector<int> result;
         while(p)
        {
            result.push_back(p->val);
            p = p->next;
        }
        return result;
    }
};
```

### 复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

遍历一次旧链表，同时创建新链表的节点，用一个Map保存旧链表和新链表的节点地址对应关系。再遍历一次旧链表，同时根据对应关系给新链表每个节点的next和random成员赋值。
本题一共需要三个辅助指针，一个用来固定指向新链表的头，一个用来遍历旧链表，一个用来遍历新链表。

```
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == nullptr)
            return pHead;
        //创建新的表头
        RandomListNode *newHead = new RandomListNode(pHead->label);
        RandomListNode *pnew = newHead;
        RandomListNode *pold = pHead->next;
        //映射表，保存新旧链表节点指针的映射关系
        map<RandomListNode *,RandomListNode *> mp;
         //第一次遍历简历映射表
        while(pold)
        {
            mp[pold] = new RandomListNode(pold->label);
            pold = pold->next;
        }
        //二次遍历就赋值操作
        pold = pHead;
        while(pold)
        {
            pnew->next = mp[pold->next];
            pnew->random = mp[pold->random];
            pold = pold->next;
            pnew = pnew->next;
        }
        return newHead;
    }
};
```

第二种思路：省去这个map，要求是仍然从老节点能直接找到对应的新节点，则方法为把新节点全都缀连在老节点之后。等设置好所有的random之后再拆分两个链表。代码如下。
注意最后新老链表都要还原否则不通过。

```
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == nullptr)
            return NULL;
        RandomListNode *p = pHead;
        RandomListNode *newNode = nullptr;
        //第一次遍历：在旧的链表上建立新的链表，插入到原链表中
        while(p)
        {
            newNode = new RandomListNode(p->label);
            newNode->next = p->next;
            p->next = newNode;
            p = p->next->next;
        }
        //第二次遍历：根据旧链表节点的值给新链表节点赋值
        p = pHead;
        while(p)
        {
            newNode = p->next;
            if(p->random == nullptr)
                newNode->random = nullptr;
            else
                newNode->random = p->random->next;
            p = p->next->next;
        }
        //拆分链表
        p = pHead;
        newNode = p->next;
        RandomListNode *newHead = newNode;
        while(p)
        {
            newNode = p->next;
            p->next = newNode->next;
            if(newNode->next)
                newNode->next = newNode->next->next;
            p = p->next;
        }
        return newHead;
    }
};
```

### 二叉搜索树与双向链表

* 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

1 借助辅助数组中序遍历的二叉搜索树的节点指针，调整它们的指针指向

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == nullptr)
            return pRootOfTree;
        //借助辅助数组和辅助栈
        vector<TreeNode *> vec;
        stack<TreeNode *> stk;
        //中序遍历，先将每个节点存于vec，再修改指针
        TreeNode *p = pRootOfTree;
        while(p || !stk.empty())
        {
            //先进入最左边
            if(p)
            {
                stk.push(p);
                p = p->left;
            }
            else
            {
                p = stk.top();
                stk.pop();
                vec.push_back(p);
                //看看有没有右子树
                p = p->right;
            }
        }
        //修改指针
        for(int i = 0; i < vec.size()-1; i++)
        {
            vec[i]->right = vec[i+1];
            vec[i+1]->left = vec[i];
        }
        
        return vec.front();
    }
};
```

递归1

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Traveral(TreeNode* root,vector<TreeNode*> &vec)
    {
        if(!root)
            return;
        Traveral(root->left, vec);
        vec.push_back(root);
        Traveral(root->right, vec);
    }
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(!pRootOfTree)
            return pRootOfTree;
        vector<TreeNode*> vec;
        stack<TreeNode*> stk;
        TreeNode* p = pRootOfTree;
        Traveral(pRootOfTree, vec);
        for(auto i = 0; i < vec.size()-1; i++)
        {
            vec[i]->right = vec[i+1];
            vec[i+1]->left = vec[i];
        }
        return vec.front();
    }
};
```

递归2

BST中序遍历就是顺序的这个大家都知道，因此在中序遍历的过程中对left和right指针进行调整，正好双向链表每个节点也是两个指针。因此不用再申请节点。调整针对的是当前节点和上一次访问的节点，这样能保证有序。 在调整������程中需要注意的是用left还是用right去指向前面的或当前的节点。因为对于中序访问，对当前节点操作时它的right节点还未访问，所以不能用right去指向之前的节点。其他的就比较简单了，代码如下

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    //需要一个节点记住上一个访问的节点
    TreeNode* pre = nullptr;
    void inOrder(TreeNode *root)
    {
        //递归结束条件就是root == nullptr
        if(root == nullptr)
            return;
        inOrder(root->left);
        //调整当前节点的左指针
        root->left = pre; 
        //调整上一个节点右指针
        if(pre)
            pre->right = root;
        pre = root;
        inOrder(root->right);
    }
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == nullptr)
            return pRootOfTree;
        TreeNode *p = pRootOfTree;
        
        //先走最左边
        while(p && p->left)
            p = p->left;
        //递归调整指针
        inOrder(pRootOfTree);
        return p;
    }
};
```

### 两个链表的第一个公共节点

输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

这里先假设链表A头结点与结点8的长度 与 链表B头结点与结点8的长度相等，那么就可以用双指针。

初始化：指针ta指向链表A头结点，指针tb指向链表B头结点
如果ta == tb， 说明找到了第一个公共的头结点，直接返回即可。
否则，ta != tb，则++ta，++tb
所以现在的问题就变成，如何让本来长度不相等的变为相等的？
假设链表A长度为a， 链表B的长度为b，此时a != b
但是，a+b == b+a
因此，可以让a+b作为链表A的新长度，b+a作为链表B的新长度

```
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        if(pHead1 == nullptr || pHead2 == nullptr)
            return nullptr;             
        ListNode *p1 = pHead1;
        ListNode *p2 = pHead2;
        //两边走相同的长度，走到第一个公共节点则退出
        while(p1 != p2)
        {
            p1 = p1 ? p1->next : pHead2;
            p2 = p2 ? p2->next : pHead1;
        }
        if(p1 == p2)
            return p1;
        return nullptr;
    }
};
```

### 链表中环的入口结点

1 哈希表

遍历单链表的每个结点
如果当前结点地址没有出现在set中，则存入set中
否则，出现在set中，则当前结点就是环的入口结点
整个单链表遍历完，若没出现在set中，则不存在环

```
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == nullptr)
            return pHead;
        set<ListNode *> hashSet;
        
        while(pHead)
        {
            //判断是否已经出现该结点
            if(hashSet.find(pHead) == hashSet.end())
            {
                hashSet.insert(pHead);
                pHead = pHead->next;
            }
            //遇到第一个重复即是入口结点
            else
            {
                return pHead;
            }
        }
        //找不到则返回nullptr
       return nullptr;
    }
};
```

1、设快慢指针，当快慢指针相遇时，快指针走的长度刚好是慢指针的两倍

2、快慢指针一定在环中相遇，而且第一次相遇慢指针一定还没绕环超过一圈，因为当慢指针进入环时，此时快指针无论在环中的哪个位置，都可以在慢指针走一圈之内追上。

3、可以设快慢指针在A点相遇，那么慢指针走过的长度位x1+x2,而快指针走过的长度为x1+x2+nL,其中L为圆环的长度,n为整数。当快慢指针相遇时，快指针走过的长度刚好是慢指针的两倍，则2(x1+x2)=x1+x2+nL,也就是x1+x2=nL。当快慢指针相遇在点A时，让新指针指向链表头和慢指针同步一步一步走，则他们刚好在入口处相遇。因为此使新指针刚好走了x1，而慢指针则刚好走了nL-x2。
```
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == nullptr)
            return pHead;
        //快慢指针
        ListNode *fast = pHead;
        ListNode *slow = pHead;
        
        //快指针每次走两步，快指针走一步，总会在环中相遇
        while(fast != nullptr && fast->next != nullptr)
        {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow)
                break;
        }
        //判断是否有环
        if(fast != nullptr &&  fast->next != nullptr)
        {
            //有环则将快指针重新设置为链表头，然后快慢指针同步走，相遇的位置就是环的入口
            fast = pHead;
            while(fast != slow)
            {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
        return nullptr;
    }
};
```

### 删除链表的重复结点

题目描述
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead == nullptr || pHead->next == nullptr)
            return pHead;
        //用一个新的头节点存储，防止 重复结点在开头
        ListNode *newHead = new ListNode(-1);
        newHead->next = pHead;
        ListNode *prev = newHead;
        ListNode *cur = pHead;
        //遍历链表
        while(cur)
        {
            //如果重复链表够长且遇到重复结点
            if(cur->next != nullptr && cur->val == cur->next->val)
            {
                //跳过当前的重复结点
                cur = cur->next;
                //将中间的重复结点都跳过
                while(cur->next && cur->val == cur->next->val)
                {
                    cur = cur->next;
                }
                //跳过最后的重复的结点，到一个新的非重复结点
                cur = cur->next;
                //修改指针指向，删除重复的结点
                prev->next = cur;
            }
            //没有重复结点，则直接到下一个结点
            else
            {
                prev = cur;
                cur = cur->next;
            }
        }
        return newHead->next;
    }
};
```

## 树

### 重建二叉树

题目描述

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```

思路1

1 由先序序列第一个pre[0]在中序序列中找到根节点位置gen

2 以gen为中心遍历
    0~gen左子树
        子中序序列：0~gen-1，放入vin_left[]
        子先序序列：1~gen放入pre_left[]，+1可以看图，因为头部有根节点
    gen+1~vinlen为右子树
        子中序序列：gen+1 ~ vinlen-1放入vin_right[]
        子先序序列：gen+1 ~ vinlen-1放入pre_right[]
 
由先序序列pre[0]创建根节点

3 连接左子树，按照左子树子序列递归（pre_left[]和vin_left[]）

4 连接右子树，按照右子树子序列递归（pre_right[]和vin_right[]）

5 返回根节点

递归方式1：用数组存储子树结果，用子数组作为参数去递归

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(pre.empty() || vin.empty() || pre.size() != vin.size())
            return nullptr;
        int preLen = pre.size();
        int vinLen = vin.size();
        //用pre的第一个元素去构造当前结点
        TreeNode *head = new TreeNode(pre[0]);
        //获取vin的分界点
        int gen = 0;
        for(int i = 0; i < vinLen; i++)
        {
            if(pre[0] == vin[i])
            {
                gen = i;
                break;
            }
        }
        //获取左子树的前序列（1，gen）和中序（0，gen-1)
        vector<int> preLeft;
        vector<int> vinLeft;
        for(int i = 0; i < gen; i++)
        {
            preLeft.push_back(pre[i+1]);
            vinLeft.push_back(vin[i]);
        }
        //获取右子树的前序列（gen+1,end）和中序列(gen+1,end)
        vector<int> preRignt;
        vector<int> vinRight;
        for(int i = gen+1; i < vinLen; i++)
        {
            preRignt.push_back(pre[i]);
            vinRight.push_back(vin[i]);
        }
        //递归
        head->left = reConstructBinaryTree(preLeft, vinLeft);
        head->right = reConstructBinaryTree(preRignt, vinRight);
        //返回
        return head;
    }
};


递归范式2：直接递归

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode *binaryTree(vector<int> &pre,vector<int> &vin, int  preBegin, int preEnd,int vinBegin, int vinEnd)
    {
        //结束条件：序列为空
        if(preBegin > preEnd || vinBegin > vinEnd)
            return nullptr;
        //找到中序的分界点在子vin的位置
        int middle = 0;
         for(auto i :vin)
        {
            if(i != pre[preBegin])
                middle++;
            else
                break;
        }
        //创建根结点
        TreeNode *root = new TreeNode(pre[preBegin]);
        //左子树
        root->left = binaryTree(pre, vin, preBegin+1 , preBegin+(middle-vinBegin), vinBegin, middle-1);
        //右子树
        root->right = binaryTree(pre, vin, preEnd-(vinEnd-(middle+1)), preEnd, middle+1, vinEnd);
        //返回 
        return root;
    }
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
       if(pre.empty() || vin.empty() || pre.size() != vin.size())
           return nullptr;
        int preBegin = 0, preEnd = pre.size()-1;
        int vinBegin = 0, vinEnd = vin.size()-1;
        return binaryTree(pre, vin, preBegin, preEnd, vinBegin, vinEnd);
    }
};
```

### 树的子结构

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

1 递归

子结构定义：树A和树B的根结点相等，并且树A的左子树和树B的左子树相等，树A的右子树和树B的右子树相等

方法：递归求解
第一步：
根据题意可知，需要一个函数判断树A和树B是否有相同的结构。显然是个递归程序。可考察递归程序3部曲。

递归函数的功能：判断2个数是否有相同的结构，如果相同，返回true，否则返回false
递归终止条件：
如果树B为空，返回true，此时，不管树A是否为空，都为true
否则，如果树B不为空，但是树A为空，返回false，此时B还没空但A空了，显然false

下一步递归参数：
如果A的根节点和B的根节点不相等，直接返回false
否则，相等，就继续判断A的左子树和B的左子树，A的右子树和B的右子树
代码
bool dfs(TreeNode *r1, TreeNode *r2) {
    if (!r2) return true;
    if (!r1) return false;
    return r1->val==r2->val && dfs(r1->left, r2->left) && dfs(r1->right, r2->right);
}

第二步：
有了上面那个函数，接下来就应该让树A的每个节点作为根节点来和B树进行比较。
遍历树A的每个节点，可用遍历算法。这里采用先序遍历。
先序遍历的模板：

void preOrder(TreeNode *r) {
    if (!r) return;
    // process r
    preOrder(r->left);
    preOrder(r->right);
}

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool compare(TreeNode *root1, TreeNode *root2)
    {
          //b树为空，不管a子树 是否为空都返回true
        if(root2 == nullptr)
            return true;
        //b树非空，而A树为空
        if(root1 == nullptr)
            return false;
        return (root1->val == root2->val) && compare(root1->left, root2->left)
            && compare(root1->right, root2->right);
    }
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
      if(pRoot1 == nullptr || pRoot2 == nullptr)
          return false;
      return compare(pRoot1,pRoot2)|| HasSubtree(pRoot1->left,pRoot2)
          || HasSubtree(pRoot1->right,pRoot2);
    }
};
```

### 二叉树的镜像（二叉树反转）

操作给定的二叉树，将其变换为源二叉树的镜像。
输入描述:
二叉树的镜像定义 ：源二叉树
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5

题目抽象：给定一颗二叉树，将二叉树的左右孩子进行翻转，左右孩子的子树做相同的操作。

方法一：递归版本
根据题意，如果我们知道一个根节点的左孩子指针和右孩子指针，那么再改变根节点的指向即可解决问题。
也就是，需要先知道左右孩子指针，再处理根节点。显然对应遍历方式中的后序遍历。
后序遍历的模板：

void postOrder(TreeNode *root) {
    if (!root) return;
    postOrder(root->left); // left child
    postOrder(root->right); // right child
    // process root
}

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* dfs(TreeNode *root)
    {
        if(root == nullptr)
            return root;
        //递归获取左右子树的指针
        TreeNode *mirrorLeft = dfs(root->left);
        TreeNode *mirrorRight = dfs(root->right);  
        //交换左右孩子的指针
        root->left = mirrorRight;
        root->right = mirrorLeft;
        
        return root;
    }
    void Mirror(TreeNode *pRoot) {
        if(pRoot == nullptr)
            return;
        dfs(pRoot);
    }
};
```

时间复杂度：O(n),n为树节点的个数。每个节点只用遍历一次，所以为O(n)
空间复杂度：O(n), 每个节点都会在递归栈中存一次

2 非递归： 层序遍历

方法一种的递归版本中遍历树的方法用的是后序遍历。所以非递归版本，只需要模拟一次树遍历。
这里模拟树的层次遍历。
层次遍历的模板为：

void bfs(TreeNode *root) {
    queue<TreeNode*> pq;
    pq.push(root);
    while (!pq.empty()) {
        int sz = pq.size();
        while (sz--) {
            TreeNode *node = pq.front(); pq.pop();
            // process node， ours tasks
            // push value to queue
            if (node->left) pq.push(node->left);
            if (node->right) pq.push(node->right);
        } // end inner while
    } // end outer while
}

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        queue<TreeNode*> que;
        if(pRoot == nullptr)    //记得判断空指针，会产生访问非法内存等未定义行为
            return;
        que.push(pRoot);
        while(!que.empty()){
            int size = que.size();
            while(size--) {
                TreeNode* p = que.front();
                que.pop();

                if(p->left)
                    que.push(p->left);
                if(p->right)
                    que.push(p->right);

                TreeNode* temp = p->left;
                p->left = p->right;
                p->right = temp;
            }
        }
    }
};
```

### 从上往下打印二叉树（层序）

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

方法：层次遍历
这道题就是一个模板题，对队列的使用。因为要满足先进先出的特性。

初始化：一个队列queue<TreeNode*> q， 将root节点入队列q
如果队列不空，做如下操作：
弹出队列头，保存为node，将node的左右非空孩子加入队列
做2,3步骤，知道队列为空
如果不需要确定当前遍历到了哪一层，模板如下：

void bfs() {
    vis[] = 0;
    queue<int> pq(start_val);
 
    while (!pq.empty()) {
        int cur = pq.front(); pq.pop();
        for (遍历cur所有的相邻节点nex) {
            if (nex节点有效 && vis[nex]==0){
                vis[nex] = 1;
                pq.push(nex)
            }
        }
    }
}

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        if(!root)
            return {};
        //辅助队列
        deque<TreeNode*> que;
        vector<int> ret;
        que.push_back(root);
        //遍历
        while(!que.empty()){
            //获取该层大小
            int size = que.size();
            while(size--) {
                TreeNode* p = que.front();
                que.pop_front();
                ret.push_back(p->val);
                if(p->left)
                    que.push_back(p->left);
                if(p->right)
                    que.push_back(p->right);
            }
        }
        return ret;
    }
};
```

### 二叉搜索树的后续遍历（二叉排序树）

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则返回true,否则返回false。假设输入的数组的任意两个数字都互不相同。

分析

这道题的解题突破点就在于二叉树的后序遍历数组有着什么的特点？

特点：遍历的时候，如果遇到比最后一个元素大的节点，就说明它的前面都比最后一个元素小，该元素后面的所有值都必须大于最后一个值，这两个条件必须都要满足。否则就说明该序列不是二叉树后序遍历。
例子： 2 4 3 6 8 7 5 这是一个正确的后序遍历
这个例子的特点就是:最后一个元素是 5 ，首先遍历数组，当遍历到6的时候，6前面的值都小于5，如果在6后面的值有一个小于5就不是后序遍历，所以一旦在遍历的时候遇到比最后一个元素的值索引，那么之后的所有元素都必须大于5，否则就不是后序遍历序列。

1 递归

```
1.分析：一次遍历确定出左右子树的分界点，然后再分别对两棵子树进行递归判断。
2.代码

class Solution {
public:
    bool isBBST(const vector<int> &sequence,int begin,int end)
    {
        //二叉树搜索树的特点：左 < 根 < 右
        //后续遍历的顺序 左，右 根
        
        //递归结束条件是只剩一个节点，就是根节点
        if(begin > end)
            return true;
        //找到左右子树分界点，即第一个比根节点大的数
        int pivot = 0;
        for( pivot = 0;  pivot <= end;  pivot++)
        {
            if(sequence[pivot] >= sequence[end])
                break;
        }
        //判断是否异常
        if(pivot > end)
            return false;
        //遍历右子树，判断是否每个都比根节点大
        for(int i = pivot; i < end; i++)
        {
            if(sequence[i] < sequence[end])
                return false;
        }
        //递归求左右子树是否符合
        return isBBST(sequence, begin, pivot-1)
            && isBBST(sequence,pivot,end-1);
        
    }
    bool VerifySquenceOfBST(vector<int> sequence) {
        if(sequence.empty())
            return false;
        return isBBST(sequence,0,sequence.size()-1);
    }
};

3.复杂度
时间复杂度：O(n*lgn)
空间复杂度：O(lgn)
```

### 二叉树种和为某一整数的路径

输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

题目抽象：给定一颗二叉树，找出满足从根节点到叶子节点和为sun的所有路径。

分析：
前置知识：

首先清楚叶子的表示：如果节点为root, 那么当前节点为叶子节点的必要条件为!root->left && !root->right

找出路径，当然需要遍历整棵树，这里采用先序遍历，即：根节点，左子树，右子树
代码如下：

void preOrder(TreeNode *root) {
 // process root

 if (root->left) preOrder(root->left);
 if (root->right) preOrder(root->right);
}
具备了上面两个前置知识后，这里无非增加了路径和sum 和 叶子节点的判断。
递归算法三部曲：

明白递归函数的功能：FindPath(TreeNode* root,int sum)，从root节点出发，找和为sum的路径
递归终止条件：当root节点为叶子节点并且sum==root->val, 表示找到了一条符合条件的路径
下一次递归：如果左子树不空，递归左子树FindPath(root->left, sum - root->val),如果右子树不空，递归右子树，FindPath(root->right, sum - root->val)

递归

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int> > ret;
        vector<int> path;
        if(root == nullptr)
            return ret;
        dfs(root,expectNumber,path,ret);
        return ret;
    }
    void dfs(TreeNode *root,int sum,vector<int> &path, vector<vector<int> >  &ret)
    {
        //递归结束条件，sum == root->val,且左右子树为nullptr(叶子)
        path.push_back(root->val);
        //满足结果的路径
        if(sum == root->val && root->left == nullptr && root->right == nullptr)
            ret.push_back(path);
        //还有左孩子
        if(root->left)
            dfs(root->left,sum-root->val,path,ret);
        //还有右孩子
        if(root->right)
            dfs(root->right,sum-root->val,path,ret);
        //表明要回溯，代表当前path中的root节点我已经不需要了
        path.pop_back();
    }
};
```

### 二叉树的深度

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

层序遍历

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == nullptr)
            return 0;
        int depth = 0;
        //辅助队列
        queue<TreeNode *> que;
        que.push(pRoot);
        while(!que.empty())
        {
            //取出该层的节点
            int size = que.size();
            while(size--)
            {
                TreeNode *p = que.front();
                que.pop();
                if(p->left)
                    que.push(p->left);
                if(p->right)
                    que.push(p->right);
            }
            depth++;
        }
        return depth;
    }
};
```

递归分治法

1 递归的目的：求深度
2 递归结束条件，叶子结点（左右孩子为空）
3 递归范围缩小

先求左边规模大约为n/2的问题，再求右边规模大约为n/2的问题，然后合并左边，右边的解

```
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        //递归的结束条件
        if(pRoot == nullptr)
            return 0;
        //获取左右子树长度
        int ldepth = TreeDepth(pRoot->left);
        int rdepth = TreeDepth(pRoot->right);
        //返回两者中较长的一边
        return  ldepth > rdepth ? ldepth +1 : rdepth+1;
    }
};
```

### 平衡二叉树

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树

平衡二叉树是左子树的高度与右子树的高度差的绝对值小于等于1，同样左子树是平衡二叉树，右子树为平衡二叉树。

每个结点为根的树的高度，然后再根据左右子树高度差绝对值小于等于1，,就可以判断以每个结点为根的树是否满足定义。

从上往下

用hash<TreeNode*, int>来存以每个结点的树的高度
再用先序遍历：根节点、左子树、右子树来判断以每个结点为根的树是否满足条件。

```
class Solution {
public:
    //缓存对应节点的深度
    map<TreeNode *, int> mp;
    //根据节点计算深度
    int depth(TreeNode *root)
    {
        if(root == nullptr)
            return 0;
        if(mp.find(root) != mp.end())
            return mp[root];
        //求左右子树的深度
        int ldepth = depth(root->left);
        int rdepth = depth(root->right);
        mp[root] = ldepth > rdepth ? ldepth+1 : rdepth+1;
        return mp[root];
    }
    //判断是否平衡
    bool isBBST(TreeNode *root)
    {
        if(root == nullptr)
            return true;
        //获取深度差
        int depth = abs(mp[root->left]-mp[root->right]);
        return depth <= 1 && isBBST(root->left) && isBBST(root->right);
    }
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(pRoot == nullptr)
            return true;
        depth(pRoot);
        return isBBST(pRoot);
    }
};
```

### 二叉树的下一个结点

题目描述
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

1 暴力

思路：

1 根据给出的结点 求出整棵树 的根结点
2 根据根结点递归求出数的中序遍历结果，存入vector
3 在vector中找到该结点，如果有下一个结点则返回下一个结点

```
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    vector<TreeLinkNode *> inOrderTraversal(TreeLinkNode *root)
    {
        if(root == nullptr)
            return {};
        stack<TreeLinkNode *> stk;
        vector<TreeLinkNode  *> res;
        if(root != nullptr)
            stk.push(root);
        while(!stk.empty())
        {
            //获取栈顶元素
            TreeLinkNode *t = stk.top();
            stk.pop();
            if(t)
            {
                //右节点压栈
                if(t->right)
                    stk.push(t->right);
                //当前结点重新压栈，左节点之后处理（访问值）
                stk.push(t);
                //nullptr跟随t插入，标识已经访问过，还没有被处理
                stk.push(nullptr);
                //左节点压栈
                if(t->left)
                    stk.push(t->left);
            }
            else
            {
                res.push_back(stk.top());
                stk.pop();
            }
        }
       return res;
    }
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        TreeLinkNode *root = nullptr;
        TreeLinkNode *temp = pNode;
        //先找到根结点
        while(temp)
        {
            root = temp;
            temp = temp->next;
        }
        //获取中序遍历结果
        vector<TreeLinkNode *> res = inOrderTraversal(root);
        //遍历找到当前结点
        for(int i = 0; i < res.size(); i++)
        {
            if(res[i] == pNode && i+1 != res.size())
                return res[i+1];
        }
        return nullptr;
    }
};
```

2 最优解法
            6
        2         7
    1       5
         3
            4

此时，可以总结一下：
[1] 是一类：特点：当前结点是父亲结点的左孩子
[2 3 6] 是一类，特点：当前结点右孩子结点，那么下一节点就是：右孩子结点的最左孩子结点,如果右孩子结点没有左孩子就是自己
[4 5]是一类，特点：当前结点为父亲结点的右孩子结点，本质还是[1]那一类
[7]是一类，特点：最尾结点


总的来说分为3类

1 当前结点作为当前路径的右孩子且有自己的右孩子，下一个结点时当前结点的右孩子的最左孩子（右孩子没有左孩子则就是右孩子它自己了）

2 当前结点作为当前路径的最左孩子（当前路径的右孩子但是右孩子没有右孩子），下一个结点是当前当前结点的父节点

3 最尾结点

```
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if(pNode == nullptr)
            return pNode;
        //下一个结点时当前结点的右孩子的最左孩子（右孩子没有左孩子则就是右孩子它自己了）
        if(pNode->right)
        {
            pNode = pNode->right;
            //一直走到最左孩子
            while(pNode->left)
            {
                pNode = pNode->left;
            }
            return pNode;
        }
        //下一个结点是当前当前结点的父节点
        while(pNode->next)
        {
            TreeLinkNode *father = pNode->next;
            if(father->left == pNode)
                return father;
            pNode = pNode->next;
        }
        //当前结点作为尾巴结点，下一个结点为空
        return nullptr;
        
    }
};
```

### 对称的二叉树

题目描述
请实现一个函数，用来判断一棵二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

递归

如果是一个对称的二叉树，那么
一个根结点的左右孩子的值是系相同的
一个非根结点的左孩子的左孩子与右孩子的右孩子对称相等；左孩子的右孩子与右孩子的左孩子对称相等
两个叶子节点同为nullptr或者值相等为对称
    只有一个为空或值不等则非对称

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot == nullptr)
            return true;
        return Equal(pRoot->left, pRoot->right);
    }

    bool Equal(TreeNode *left, TreeNode *right)
    {
        //两个叶子节点为空则对称
        if(left == nullptr && right == nullptr)
            return true;
        //两个叶子只有一个为空或者值不等则非对称
        if(left == nullptr || right == nullptr || left->val != right->val)
            return false;
        //递归比较他们子树进行比较3
        return Equal(left->left, right->right) && Equal(left->right,right->left);
    }
};
```

2 栈
对称的二叉树，它的中左右遍历结果和中右左遍历结果应该相同
用两个栈存储他们的遍历结果
然后逐个弹出比较值，遇到值不等则非对称
如果最后双栈都为空，则对称

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        //空树
        if(pRoot == nullptr)
            return true;
        //只有一个节点
        if(pRoot->left == nullptr && pRoot->right == nullptr)
            return  true;
        //单个孩子节点为空
        if(pRoot->left == nullptr ^ pRoot->right == nullptr)
            return false;
        //st1存储根节点左子树的中左右顺序遍历结果
        stack<TreeNode *> st1;
        st1.push(pRoot->left);
        //st2存储根节点右子树的中右左顺序遍历结果
        stack<TreeNode *> st2;
        st2.push(pRoot->right);
        
        //遍历
        while(!st1.empty() && !st2.empty())
        {
            TreeNode *node1 = st1.top();
            TreeNode *node2 = st2.top();
            st1.pop();
            st2.pop();
            //判断对应的顶点是否值对称
            if(node1->val != node2->val)
                return false;
            //判断左孩子的左孩子和右子树的右孩子是否单个为空
            if(node1->left == nullptr ^ node2->right == nullptr)
                return false;
            //判断左孩子的右孩子和右孩子的左孩子是否单个为空
            if(node1->right == nullptr ^  node2->left == nullptr)
                return false;
            //分别对应进栈，等下一轮判断
            //左子树的左孩子
            if(node1->left)
                st1.push(node1->left);
            //右子树的右孩子
            if(node2->right)
                st2.push(node2->right);
            //左子树的右孩子
            if(node1->right)
                st1.push(node1->right);
            //右子树的左孩子
            if(node2->left)
                st2.push(node2->left);
        }
        if(st1.empty() && st2.empty())
            return true;
        return false;
    }

};
```

### 按之字顺序打印二叉树

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

思想还是用辅助队列进行层序遍历

此题跟“按层打印二叉树”，仅有一点区别，“按层打印二叉树”是每层都按照从左到右打印二叉树。

而此题是，按照奇数层，从左到右打印，偶数层，从右到左打印。

奇数层左到右，偶数层右到左

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ret;
            if(pRoot == nullptr)
                return ret;
            //辅助队列
            queue<TreeNode *> que;
            que.push(pRoot);
            //奇数层从左到右遍历,偶数层从右到左遍历
            int  level = 0;
            while(!que.empty())
            {
                int size = que.size();
                vector<int> ans;
                while(size--)
                {
                    TreeNode *node = que.front();
                    que.pop();
                    ans.push_back(node->val);
                    if(node->left)
                        que.push(node->left);
                    if(node->right)
                        que.push(node->right);
                }
                level++;
                //如果是偶数层，反转左 ->右成 右->左
                if(!(level & 1))
                    reverse(ans.begin(),ans.end());
                ret.push_back(ans);
            }
        }
    
};
```

### 序列化二叉树

题目描述L:

请实现两个函数，分别用来序列化和反序列化二叉树

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

例如，我们可以把一个只有根节点为1的二叉树序列化为"1,"，然后通过自己的函数来解析回这个二叉树

### 把二叉树打印成多行

辅助队列进行层序遍历

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ret;
            if(pRoot == nullptr)
                return ret;
            queue<TreeNode *> que;
            que.push(pRoot);
            while(!que.empty())
            {
                int size = que.size();
                vector<int> ans;
                while(size--)
                {
                    TreeNode *node = que.front();
                    que.pop();
                    ans.push_back(node->val);
                    if(node->left)
                        que.push(node->left);
                    if(node->right)
                        que.push(node->right);
                }
                ret.push_back(ans);
            }
            return ret;
        }
};
```

### 题目描述

请实现两个函数，分别用来序列化和反序列化二叉树

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

例如，我们可以把一个只有根节点为1的二叉树序列化为"1,"，然后通过自己的函数来解析回这个二叉树

输入
{8,6,10,5,7,9,11}
返回值
{8,6,10,5,7,9,11

1 先序递归遍历

序列化的结果为字符串 str, 初始str = "".根据要求，遇到nullptr节点，str += "#"

遇到非空节点，str += "val" + "!"; 假设val为3， 就是 str += "3!"

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    char* Serialize(TreeNode *root) {    
        if(root == nullptr)
            return "#";
        //先将先序遍历结果转为string
        string res = to_string(root->val);
        res.push_back(',');
        //递归获取左右子树的结果
        char * left = Serialize(root->left);
        char *right = Serialize(root->right);
        //申请新的空间存储
        char * ret = new char[strlen(left) + strlen(right)+res.size()];
        strcpy(ret,res.c_str());
        strcat(ret,left);
        strcat(ret,right);
        
        return ret;
        
    }
    //参数使用引用&， 以实现全局变量的目的
    TreeNode * buildTree(char * &str)
    {    
        //结束条件
        if(*str == '#')
        {
              ++str;
            return nullptr;
        }
        //构造根节点
        int num = 0;
        //转为数字
        while(*str  != ',')
        {
            num = num*10 + (*str - '0');
            ++str;
        }
        //递归
        ++str;
        TreeNode *root = new TreeNode(num);
        root->left =Deserialize(str);
        root->right =Deserialize(str);
        return root;
    }
    TreeNode* Deserialize(char * &str) {
        return buildTree(str);
    }
};
```

层次遍历采用队列实现。跟先序遍历的思想差不多，无非都是把树的所有数据遍历一遍，然后记录下来。

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    char* Serialize(TreeNode *root) {    
        string s;
        queue<TreeNode *> que;
        que.push(root);
        
        while(!que.empty())
        {
            //获取队头
            TreeNode *node = que.front();
            que.pop();
            
            //空节点
            if(node == nullptr)
            {
                s.push_back('#');
                s.push_back(',');
                continue;
            }
            
            //非空
            s += to_string(node->val);
            s.push_back(',');
            
            //左右孩子进队
            que.push(node->left);
            que.push(node->right);
        }
        //转为char *
        char *ret = new char[s.length() + 1];
        strcpy(ret,s.c_str());
        return ret;
    }
   TreeNode* Deserialize(char *str)
    {
        if (str == nullptr) {
            return nullptr;
    }
        // 可用string成员函数
        string s(str);
        if (str[0] == '#') {
            return nullptr;
        }
 
        // 构造头结点
        queue<TreeNode*> nodes;
        TreeNode *ret = new TreeNode(atoi(s.c_str()));
        s = s.substr(s.find_first_of(',') + 1);
        nodes.push(ret);
        // 根据序列化字符串再层次遍历一遍，来构造树
        while (!nodes.empty() && !s.empty())
        {
            TreeNode *node = nodes.front();
            nodes.pop();
            if (s[0] == '#')
            {
                node->left = nullptr;
                s = s.substr(2);
            }
            else
            {
                node->left = new TreeNode(atoi(s.c_str()));
                nodes.push(node->left);
                s = s.substr(s.find_first_of(',') + 1);
            }
 
            if (s[0] == '#')
            {
                node->right = nullptr;
                s = s.substr(2);
            }
            else
            {
                node->right = new TreeNode(atoi(s.c_str()));
                nodes.push(node->right);
                s = s.substr(s.find_first_of(',') + 1);
            }
        }
        return ret;
    }
};
```

### 二叉树的第 k小个的节点

题目描述
给定一棵二叉搜索树，请找出其中的第k小的结点。

先序遍历

递归
size 作为全局变量，在先序遍历的过程中，每遍历一个就size--，当size == 1时，用全局变量记录下该结点指针

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    int size = 0;
    TreeNode *ans;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot == nullptr)
            return nullptr;
        ans = nullptr; //初始化
        size = k;
        dfs(pRoot);
        return ans;

    }

    void dfs(TreeNode *p)
    {
        if( p == nullptr || size < 1)
            return;
        dfs(p->left);
        if(size == 1)
            ans = p;
        if(--size > 0)
            dfs(p->right);
    }
};
```

迭代
辅助栈模拟先序遍历，迭代的方式进行先序遍历，当size== 1时返回当前结点指针即可

```
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot == nullptr)
            return pRoot;
        //辅助栈
        stack<TreeNode *> stk;
        TreeNode *p = pRoot;
        //栈非空或当前结点还有右孩子还没访问
        while(!stk.empty() || p)
        {
            //先走到最左边
            while(p)
            {
                stk.push(p);
                p = p->left;
            }
            //获取栈顶元素
            TreeNode *node = stk.top();
            stk.pop();
            //判断 是否到了第k个点
            if((--k) == 0)
                return node;
            //将右子树加入栈
            p = node->right;
        }
        return nullptr;
    }
};
```

## 字符串

### 替换空格

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

```

暴力

先转为string,然后处理完成后再转为char *。但不是以返回值的形式，还要利用好原来的空间，用strcpy实现之。

class Solution {
public:
    void replaceSpace(char *str,int length) {
      string res, s = str;
        for(char c : s){
            if(c == ' ') res += "%20";
            else res += c;
        }
        strcpy(str,res.c_str());
    }
};


逆向遍历

分析：由于函数返回为void，说明此题不能另外开辟数组，需要in-place操作。我们知道字符串的遍历无非是从左到右和从右到左两种。
1）如果从左到右，会发现如果遇到空格，会将原来的字符覆盖。于是，此方法不行。
2）那么就考虑从右向左，遇到空格，就填充“20%“，否则将原字符移动应该呆的位置。
    那么就要先计算新的字符数组的大小
    然后从后往前遍历并覆盖

class Solution {
public:
    void replaceSpace(char *str,int length) {
        if(length == 0 || str == nullptr)
            return;
        int cnt = 0; //统计空格个数，确定扩充后的总的长度
        for(int i = 0; i != length; ++i)
            if(str[i] == ' ') ++cnt;
        int newLength = length + cnt * 2;
        //从后往前修改
        for(int i = length; i >= 0; --i) {
            if(str[i] == ' ') {
                str[newLength--] = '0';
                str[newLength--] = '2';
                str[newLength--] = '%';
            }
            else
                str[newLength--] = str[i];
        }
    }
};
```

### 字符串的排列

题目描述
输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

递归三部曲：

递归函数的功能：dfs(int pos, string s), 表示固定字符串s的pos下标的字符s[pos]
递归终止条件：当pos+1 == s.length()的时候，终止，表示对最后一个字符进行固定，也就说明，完成了一次全排列
下一次递归：dfs(pos+1, s), 很显然，下一次递归就是对字符串的下一个下标进行固定

```
class Solution {
public:
    vector<string> Permutation(string str) {
        if(str.empty())
            return {};
        set<string> ret;
        perm(0,str,ret);
        return vector<string> ({ret.begin(),ret.end()});
    }
    void perm(int position,string s, set<string> &ret)
    {
        //结束条件
        if(position+1 == s.length())
        {
            ret.insert(s);
        }

        for(int i = position; i < s.length(); ++i)
        {
            //先交换
            swap(s[position],s[i]);
            perm(position+1,s,ret);
            //回溯
            swap(s[position],s[i]);
        }
    }
};
```

### 第一个只出现一次的字符

1 哈希

```
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        if(str.empty())
            return -1;
        map<char,int> mp;
        for(char c :str)
        {
            mp[c]++;
        }
        for(int i = 0; i <  str.size(); i++)
        {
            if(mp[str[i]] == 1)
                return i;
        }
        return -1;
    }
};
```

2 bitset
bitset存储二进制数位。

bitset就像一个bool类型的数组一样，但是有空间优化——bitset中的一个元素一般只占1 bit，相当于一个char元素所占空间的八分之一。

bitset中的每个元素都能单独被访问，例如对于一个叫做foo的bitset，表达式foo[3]访问了它的第4个元素，就像数组一样。

bitset有一个特性：整数类型和布尔数组都能转化成bitset。

bitset的大小在编译时就需要确定。如果你想要不确定长度的bitset，请使用（奇葩的）vector<bool>。

```
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        bitset<128> b1, b2;
        for (const char ch : str) {
            if (!b1[ch] && !b2[ch]) {
                b1[ch] = 1;
            }
            else if (b1[ch] && !b2[ch]) {
                b2[ch] = 1;
            }
        }
        for (int i=0; i<str.length(); ++i) {
            if (b1[str[i]] && !b2[str[i]]) {
                return i;
            }
        }
        return -1;
    }
};
```

### 左旋转字符串

对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。

1 库函数

```
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int size = str.size();
        if(size < n)
            return str;
        return str.substr(n) + str.substr(0,n);
    }
};
```

2暴力

```
class Solution {
public:
    string LeftRotateString(string str, int n) {
        if(n > str.size())
            return str;
        string ret = "";
        for(int i = n; i < str.size(); i++)
            ret += str[i];
        for(int i = 0; i < n; i++)
            ret += str[i];
        return ret;
    }
};
```

### 翻转单词序列

例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。

1 使用库函数

```
class Solution {
public:
    string ReverseSentence(string str) {
        // 预处理
        if (str.empty()) return str;
        int i = 0, sz = str.size();
        while(i < sz && str[i] == ' ') ++i;
        if (i == sz) return str;
        istringstream ss(str);
        vector<string> ret;
        string s;
        // 拆分单词
        while (ss >> s)  
            ret.push_back(s);
        reverse(ret.begin(), ret.end());
        ostringstream oss;
        // 合并成字符串
        for (int i=0; i<ret.size()-1; ++i) 
            oss << ret[i] << ' ';
        oss << ret.back();
        return oss.str();
    }
};
```

2 迭代划分子串

```
class Solution {
public:
    string ReverseSentence(string str) {
        if(str.empty())
            return str;
        //先划分子串
        string substr;
        vector<string> ret;
        for(int i = 0; i < str.size(); i++)
        {
            if(str[i] != ' ')
                substr += str[i];
            else
            {
                ret.push_back(substr);
                substr.clear();
            }
        }
        //将最后一个子串添加
        ret.push_back(substr);
        
        //拼接结果
        string result;
        for(int i = ret.size()-1 ; i > 0; i--)
        {
            result += ret[i];
            result += ' ';
        }
        //处理第一个子串
        result += ret[0];
        return result;
    }
};
```

3 利用栈

```
class Solution {
public:
    string ReverseSentence(string str) {
        if(str.empty())
            return str;
        //先划分子串
        string substr;
        stack<string> stk;
        for(int i = 0; i < str.size(); i++)
        {
            if(str[i] != ' ')
                substr += str[i];
            else
            {
                stk.push(substr);
                substr.clear();
            }
        }
        //将最后一个子串添加
        stk.push(substr);
        
        //拼接结果
        string result;
        while(stk.size() > 1)
        {
            result += stk.top();
            result += ' ';
            stk.pop();
        }
        //处理第一个子串
        result += stk.top();
        return result;
    }
};
```

### 把字符串转换为整数

将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

int的范围
如果超过了这两个范围该怎么办？
其实也很简单，首先判断这个数的正负，如果正数，超过了INT_MAX，就设置为INT_MAX，如果是负数，首先我们不考虑负号，如果超过了INT_MAX+1, 则就置为INT_MAX+1, 最后再根据正负号，来加负号。

```
class Solution {
public:
    int StrToInt(string str) {
        //特殊情况处理
        int len = str.size();
        if(len == 0)
            return 0;
        int i = 0;
        //检查非法开头
        if(!isdigit(str[i]) && str[i] != '+'
             && str[i]  != '-')
            return 0;
        //检查符号,true 为负数
        bool flag = str[i] == '-' ? true : false;
        //如果为开头为非数组则跳过
        i = isdigit(str[i]) ? i : i+1;
        long ans = 0;
        
        //转化
        while(i < len && isdigit(str[i]))
        {
            ans = ans * 10 + (str[i]-'0');
            i++;
            //如果正数过大则跳出
            if(flag == false && ans > INT_MAX)
            {
                ans = INT_MAX;
                break;
            }
            //超过整数的负数最大值
            if(flag  && ans > 1L + INT_MAX)
            {
                ans = 1L + INT_MAX;
                break;
            }
        }
        //遇到非法情况回提前退出
        if(i < len)
            return 0;
        if(flag)
            return static_cast<int>(-ans);
        else
            return static_cast<int>(ans);
    }
};
```

### 表示数值的字符串

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

```
class Solution {
    public: bool isNumeric(char* str) {
        int n=strlen(str);

        bool point = false, exp = false; // 标志小数点和指数

        for (int i = 0; i < n; i++) {
            //遇到正负号
            if (str[i] == '+' || str[i] == '-') 
            {
                 // +-号不是最后一位（i+1 == n确保后面不越界)后面必定为数字 或 后面为.（-.123 = -0.123）
                if (i + 1 == n && !(str[i + 1] >= '0' && str[i + 1] <= '9' || str[i + 1] == '.')) 
                {
                    return false;
                }
                // +-号只出现在第一位或eE的后一位
                if (!(i == 0 || str[i-1] == 'e' || str[i-1] == 'E')) { 
                    return false;
                }

            //遇到.
            } else if (str[i] == '.')
            {
                // .后面必定为数字 或为最后一位（233. = 233.0）
                if (point || exp || !(i + 1 < n && str[i + 1] >= '0' && str[i + 1] <= '9')) { 
                    return false;
                }
                point = true;
            //遇到e
            } else if (str[i] == 'e' || str[i] == 'E') {
                // eE后面必定为数字或+-号
                //不能是最后一位
                //e/E不能重复出现
                if (exp || i + 1 == n || !(str[i + 1] >= '0' && str[i + 1] <= '9' || str[i + 1] == '+' || str[i + 1] == '-')) { 
                    return false;
                }
                exp = true;
            //遇到数字就跳过
            } else if (str[i] >= '0' && str[i] <= '9') 
            {
            //特殊字符就返回 false
            } else {
                return false;
            }
        }
        return true;
    }
};
```

### 字符串中第一个不重复的字符

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

1 借助队列和辅助 哈希表

```
#include<unordered_map>
class Solution
{
private:
    //辅助队列
    queue<char> que;
    //辅助映射表
    unordered_map<char,int> mp;
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        //如果是第一次出现，添加到队列中
         if(mp.find(ch) == mp.end())
             que.push(ch);
        //累加出现次数
         ++mp[ch];
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
       while(!que.empty())
       {
           char ch = que.front();
           //头部如果是第一次 出现，直接返回
           if(mp[ch] == 1)
           {
               return ch;
           }
            //不是第一次出现则弹出，找下一个
           else
           {
               que.pop();
           }
       }
       //找不到则返回#
       return '#';
    }
};
```

## 栈和队列

### 用两个栈实现队列

题目描述
  
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型

```
解法1：每次都是用第二个栈做中转
    stack1入栈前先把stack1的元素都移动到stack2
    stack1新元素入栈
    stack2的元素再逐个入栈到stack1

class Solution
{
public:
    void push(int node) {
        while(!stack1.empty())
        {
            stack2.push(stack1.top());
            stack1.pop();
        }
        stack1.push(node);
        while(!stack2.empty())
        {
            stack1.push(stack2.top());
            stack2.pop();
        }
    }

    int pop() {
        int num = stack1.top();
            stack1.pop();
            return num ;
    }
private:
    stack<int> stack1;
    stack<int> stack2;
};


解法2：
1，入队时，统一将元素压入栈1
2，出站时
    1 检查栈2是否为空，为空则将栈1的元素逐个出栈并压入栈2
    2 获取栈2栈顶元素，出栈操作并

class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }
    int pop() {
        if(stack2.empty())
        {
          while(!stack1.empty())
         {
             stack2.push( stack1.top());
             stack1.pop();
         }
        }
        int t = stack2.top();
        stack2.pop();
        return t;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

### 包含min函数的最小栈

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

```
class Solution {
public:
    void push(int value) {
        stack1.push(value);
        if(stack2.empty())
            stack2.push(value);
        else 
          value < stack2.top() ? stack2.push(value) : stack2.push(stack2.top());
    }
    void pop() {
        stack1.pop();
        stack2.pop();
    }
    int top() {
        return stack1.top();
    }
    int min() {
        return stack2.top();
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

### 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

思路：

题目抽象：给出一个入栈序列pushV和出栈序列popV, 判断出栈序列是否满足条件。

方法：模拟法
直接模拟即可。因为弹出之前的值都会先入栈，所以这里用个栈来辅助。

初始化：用指针i指向pushV的第一个位置， 指针j指向popV的第一个位置
如果pushV[i] != popV[j]， 那么应该将pushV[i]放入栈中， ++i
否则，pushV[i]==popV[j], 说明这个元素是放入栈中立马弹出，所以，++i, ++j，然后应该检查popV[j]
与栈顶元素是否相等，如果相等，++j, 并且弹出栈顶元素
4，重复2，3， 如果i==pushV.size(), 说明入栈序列访问完，此时检查栈是否为空，如果为空，说明匹配，斗则不匹配。

```
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        int insize = pushV.size();
        int outsize = popV.size();
        if( insize != outsize)
            return false;

        //辅助栈
        stack<int> stk;

        int i = 0, j = 0;
        while(i < insize && j < outsize)
        {
            if(pushV[i] != popV[j])
            {
                 stk.push(pushV[i]);
                i++;
            }
            else
            {
                stk.push(pushV[i]);
                stk.pop();
                i++;
                j++;
                while(!stk.empty() && popV[j] == stk.top())
                {
                    stk.pop();
                    j++;
                }
            }
        }
        if(stk.empty())
            return true;
        return false;
    }
};
```

## 其他

### 扑克牌顺子

题目描述

一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,
决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。
要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。

题目抽象：给定一个长度为5（排除空vector），包含0-13的数组，判断公差是否为1.

如果vector中不包含0的情况：
那么如何判断呢？因为需要是顺子，所以首先不能有重复值， 如果没有重复值，那么形如[1 2 3 4 5]
[5 6 7 8 9]， 会发现最大值与最小值的差值应该小于5.

二. 如果vector中包含0：
发现除去0后的值，判断方法和1中是一样的。

所以根据如上两个条件，算法过程如下：

初始化一个set，最大值max_ = 0, 最小值min_ = 14
遍历数组， 对于大于0的整数，没有在set中出现，则加入到set中，同时更新max_, min_
如果出现在了set中，直接返回false
数组遍历完，最后再判断一下最大值与最小值的差值是否小于5

```
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if (numbers.empty()) return false;
        //条件：五个牌子不重复，最大最小值差值5
        set<int> st;
        //维护最大最小值
        int maxSize = 0;
        int minSize = 14;
        for (int val : numbers) {
            if (val > 0) {
                if (st.count(val) > 0) return false;
                st.insert(val);
                maxSize = max(maxSize, val);
                minSize = min(minSize, val);
            }
        }
        return maxSize- minSize < 5;
    }
};
```

### 孩子们的游戏（圆圈最后剩下的数）

题目描述：
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此��HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

如果没有小朋友，请返回-1

1 模拟法

最开始长度为n，每次删除一个数，长度变为n-1，如果用数组模拟操作的话，删除一个数据，涉及大量的数据搬移操作，所以我们可以使用链表来模拟操作。

```
class Solution {
public:
    int LastRemaining_Solution(int n, int m)
    {
        if(n <= 0)
            return -1;
         list<int> list;
        //所有元素添加到链表
        for(int i = 0; i < n; i++)
        {
            list.push_back(i);
        }
        int index = 0;
        //逐个选出，直到剩余一个
        while(n > 1)
        {
            index = (index + m-1) %n;
            auto it = list.begin();
            //迭代器前移
            advance(it, index);
            list.erase(it);
            n--;
        }
        return list.back();
    }
};
```

2 递归法

分析：

长度为 n 的序列会先删除第 m % n 个元素，然后剩下一个长度为 n - 1 的序列。那么，我们可以递归地求解 f(n - 1, m)，就可以知道对于剩下的 n - 1 个元素，最终会留下第几个元素，我们设答案为 x = f(n - 1, m)。

由于我们删除了第 m % n 个元素，将序列的长度变为 n - 1。当我们知道了 f(n - 1, m) 对应的答案 x 之后，我们也就可以知道，长度为 n 的序列最后一个删除的元素，应当是从 m % n 开始数的第 x 个元素。因此有 f(n, m) = (m % n + x) % n = (m + x) % n。

当n等于1时，f(1,m) = 0
代码为：

```
class Solution {
public:
    //返回大的下一次开始数的位置
    int f(int n, int m) {
        if (n == 1) return 0;
        int x = f(n-1, m);
        return (x+m) % n;
    }
    int LastRemaining_Solution(int n, int m)
    {
        if (n <= 0) return -1;
        return f(n,m);
    }
};
```

3 迭代法

根据方法二可知，
f[1] = 0
f[2] = (f{1] + m) % 2
f[3] = (f[2] + m) % 3
...
f[n] = (f[n-1] + m) % n

```
class Solution {
public:

    int LastRemaining_Solution(int n, int m)
    {
        if (n <= 0) return -1; 
        int index = 0;
        for (int i=2; i<=n; ++i) {
            index = (index + m) % i;
        }
        return index;
    }
};
```

### 数组中重复的数字

* 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中第一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

1 哈希表

```
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        if(length < 2)
            return false;
        vector<bool> f(length,false);
        //遍历
        for(int i = 0; i < length; i++)
        {
            if(f[numbers[i]]== false)
                f[numbers[i]] = true;
            //遇到重复元素
            else
            {
                *duplication = numbers[i];
                return true;
            }
        }
        return false;
    }
};
```

### 构建乘积数组（除了自身以外的乘积）

题目：
题目描述：给定一个长度为n的数组A，求数组B，B[i] = A[0]A[1]...A[i-1]*A[i+1]...A[n-1]。
要求不能使用除法。

即：除了自身外的数组的乘积，可以划分为左右（前缀和后缀）相乘得到答案

方法一：左右乘积列表
对于给定索引 i，我们将使用它左边所有数字的乘积乘以右边所有数字的乘积

对于索引i A[i]
i左侧所有数字的乘积
    left[i] = A[0]*...*A[i-1]
i右侧的所有元素的乘积
    right[i] = A[i+1]*...*A[n-1]

所以：
B[i] = left[i] * right[i]

同理
left[i+1] = A[0]*...A[i-1]*A[i]
right[i+1] = A[i+2]*...*A[n-1]

于是，
left[i+1] = left[i] * A[i]
right[i] = right[i+1] * A[i+1]

```
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int size = A.size();
        //L和R代表左右两侧的乘积列表
        vector<int> L(size,0),R(size,0);
        //结果集合
        vector<int> B(size);
        //先求左侧的乘积和
        //左侧没元素，对于索引0要特殊处理设置为1
        L[0] = 1;
        for (int i=1; i<size; ++i) {
            L[i] = L[i-1] * A[i-1];
        }

        //R[i]为索引i右侧的所有元素的乘积
        //对于索引size-1的元素，右侧没有元素，值为1
        R[size-1] = 1;
        for(int i = size-2; i >= 0; i--)
        {
            R[i] = R[i+1] * A[i+1];
        }
        
        //对于索引i，除 A[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for(int i = 0; i < size; i++)
        {
            B[i] = L[i] * R[i];
        }
        return B;
    }
};
```
复杂度分析

时间复杂度：O(N)O(N)，其中 NN 指的是数组 nums 的大小。预处理 L 和 R 数组以及最后的遍历计算都是 O(N)O(N) 的时间复杂度。
空间复杂度：O(N)O(N)，其中 NN 指的是数组 nums 的大小。使用了 L 和 R 数组去构造答案，L 和 R 数组的长度为数组 nums 的大小。

2 优化：空间复杂度为o(1)

由于输出数组不算在空间复杂度内，那么我们可以将 L 或 R 数组用输出数组来计算。先把输出数组当作 L 数组来计算，然后再动态构造 R 数组得到结果。让我们来看看基于这个思想的算法。

算法

1 初始化 answer 数组，对于给定索引 i，answer[i] 代表的是 i 左侧所有数字的乘积。
2 构造方式与之前相同，只是我们试图节省空间，先把 answer 作为方法一的 L 数组。
3 这种方法的唯一变化就是我们没有构造 R 数组。而是用一个遍历来跟踪右边元素的乘积。并更新数组 answer[i]=answer[i]*Ranswer[i]=answer[i]∗R。然后 RR 更新为 R=R*nums[i]R=R∗nums[i]，其中变量 RR 表示的就是索引右侧数字的乘积

```
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int size = A.size();
        //结果集合
        vector<int> B(size);
        //B[i] 表示索引 i 左侧所有元素的乘积
        //左侧没元素，对于索引0要特殊处理设置为1
        B[0] = 1;
        for (int i=1; i<size; ++i) {
            B[i] = B[i-1] * A[i-1];
        }

        //R[i]为索引i右侧的所有元素的乘积
        //对于索引size-1的元素，右侧没有元素，值为1
        int R = 1;
        for(int i = size-1; i >= 0; i--)
        {
            B[i] = B[i] * R;
            R *=  A[i];
        }
        
        return B;
    }
};
```

### 正则表达式匹配

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

1 递归

分析：

假设主串为s，长度为sn， 模式串为p，长度为pn，对于模式串p当前的第i位来说，有'正常字符'、'*'、'.'三种情况。我们针对这三种情况进行讨论：

1 如果p[i]为正常字符， 那么我们看s[i]是否等于p[i], 如果相等，说明第i位匹配成功,接下来看s[i+1...sn-1] 和 p[i+1...pn-1]

2 如果p[i] 为'.', 它能匹配任意字符，直接看s[i+1...sn-1] 和 p[i+1...pn-1]

3 如果p[i] 为'*'， 表明p[i-1]可以重复0次或者多次，需要把p[i-1] 和 p[i]看成一个整体.

如果p[i-1]重复0次，则直接看s[i...sn-1] 和 p[i+2...pn-1]
如果p[i-1]重复一次或者多次,则直接看s[i+1...sn-1] 和p[i...pn-1]，但是有个前提：s[i]==p[i] 或者 p[i] == '.'

细化

显然上述的过程可以递归进行计算。
则递归三部曲为：

递归函数功能：match(s, p) -> bool, 表示p是否可以匹配s

递归终止条件：

如果s 和 p 同时为空，表明正确匹配
如果s不为空，p为空，表明，不能正确匹配
如果s为空，p不为空，需要计算，不能直接给出结果
下一步递归：

对于前面讨论的情况1，2进行合并，如果*s == *p || *p == '.',则match(s+1, p+1)

对于情况3，如果重复一次或者多次，则match(s+1,p),如果重复0次，则match(s, p+2)

/*
    解这题需要把题意仔细研究清楚，反正我试了好多次才明白的。
    首先，考虑特殊情况：
         1>两个字符串都为空，返回true
         2>当第一个字符串不空，而第二个字符串空了，返回false（因为这样，就无法
            匹配成功了,而如果第一个字符串空了，第二个字符串非空，还是可能匹配成
            功的，比如第二个字符串是“a*a*a*a*”,由于‘*’之前的元素可以出现0次，
            所以有可能匹配成功）
    之后就开始匹配第一个字符，这里有两种可能：匹配成功或匹配失败。但考虑到pattern
    下一个字符可能是‘*’， 这里我们分两种情况讨论：pattern下一个字符为‘*’或
    不为‘*’：
          1>pattern下一个字符不为‘*’：这种情况比较简单，直接匹配当前字符。如果
            匹配成功，继续匹配下一个；如果匹配失败，直接返回false。注意这里的
            “匹配成功”，除了两个字符相同的情况外，还有一种情况，就是pattern的
            当前字符为‘.’,同时str的当前字符不为‘\0’。
          2>pattern下一个字符为‘*’时，稍微复杂一些，因为‘*’可以代表0个或多个。
            这里把这些情况都考虑到：
               a>当‘*’匹配0个字符时，str当前字符不变，pattern当前字符后移两位，
                跳过这个‘*’符号；
               b>当‘*’匹配1个或多个时，str当前字符移向下一个，pattern当前字符
                不变。（这里匹配1个或多个可以看成一种情况，因为：当匹配一个时，
                由于str移到了下一个字符，而pattern字符不变，就回到了上边的情况a；
                当匹配多于一个字符时，相当于从str的下一个字符继续开始匹配）
    之后再写代码就很简单了。
*/

```
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        //递归终止条件是str和pattern都为空
        if(*str == '\0' && *pattern == '\0')
            return true;
        //如果str非空，而pattern 为空
        if(*pattern == '\0')
            return false;
        //剩余的情况是str为空(非空)，pattern非空
        
        //如果下一个字符不是 *
        if(*(pattern+1) != '*')
        {
            //情况1/2, pattern 与 str字符相同或者*pattern == '.'
            if(*str == *pattern || (*pattern == '.' && *str != '\0'))
                return match(str+1, pattern+1);
            else
                return false;
        }
        else
        {
            //当*匹配到1或多个字符，str向前移动一个，而pattern当前字符不变
            if(*str == *pattern || (*str != '\0' && *pattern == '.'))
                return match(str,pattern+2) || match(str+1,pattern);
            //当 * 匹配到0个字符，str当前字符不变，pattern向后移动两位，跳过*
            else
                return match(str, pattern+2);
        }
    }
};
```

### 数据流中的中位数

题目描述
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

暴力

```
class Solution {
public:
    vector<int> v;
    void Insert(int num)
    {
        v.push_back(num);

    }

    double GetMedian()
    { 
        sort(v.begin(), v.end());
        int size = v.size();
        if (size & 1) {
            return (double)v[size >> 1];
        }
        else {
            return (v[szie >> 1] + v[(size - 1) >> 1]) / 2.0;
        }
    }

};
```

1 插入排序

```
class Solution {
public:
    vector<int> v;
    void Insert(int num)
    {
        if(v.empty()){
            v.push_back(num);
        }else{
            auto it = lower_bound(v.begin(), v.end(), num);
            v.insert(it, num);
        }
    }

    double GetMedian()
    { 
        int len = v.size();
        if(len&1){
            return v[len >> 1];
        }else{
            return double((v[(len-1) >> 1] + v[len >> 1])/2.0);
        }
    }

};
```

2 堆排序

假设[0 ... median - 1]的长度为l_len, [median + 1 ... arr.sise() - 1]的长度为 r_len.
1.如果l_len == r_len + 1, 说明，中位数是左边数据结构的最大值
2.如果l_len + 1 == r_len, 说明，中位数是右边数据结构的最小值
3.如果l_len == r_len, 说明，中位数是左边数据结构的最大值与右边数据结构的最小值的平均值。

```
class Solution {
public:
    #define SCD static_cast<double>
    priority_queue<int> min_q; // 大顶推
    priority_queue<int, vector<int>, greater<int>> max_q; // 小顶堆

    void Insert(int num)
    {

        min_q.push(num); // 试图加入到大顶推

        // 平衡一个两个堆
        max_q.push(min_q.top()); 
        min_q.pop();

        if (min_q.size() < max_q.si***_q.push(max_q.top());
            max_q.pop();
        }

    }

    double GetMedian()
    { 
        return min_q.size() > max_q.size() ? SCD(min_q.top()) : SCD(min_q.top() + max_q.top()) / 2;
    }

};
```

### 矩阵中的路径

题目描述：给定一个二维字符串矩阵mat,和一个字符串str,判断str是否可以在mat中匹配。
可以选择的方向是上下左右。

dfs模板
```
dfs(){
 
    // 第一步，检查下标是否满足条件
 
    // 第二步：检查是否被访问过，或者是否满足当前匹配条件
 
    // 第三步：检查是否满足返回结果条件
 
    // 第四步：都没有返回，说明应该进行下一步递归
    // 标记
    dfs(下一次)
    // 回溯
} 
 
 
main() {
    for (对所有可能情况) {
        dfs()
    }
}
```

```
class Solution {
public:
    char *mat = 0;
    int h = 0, w = 0;
    int str_len = 0;
    int dir[5] = {-1, 0, 1, 0, -1};

    //深度遍历递归函数
    bool dfs(int i, int j, int pos, char *str) {
        // 因为dfs调用前，没有进行边界检查，
        // 所以需要第一步进行边界检查，
        // 因为后面需要访问mat中元素，不能越界访问
        if (i < 0 || i >= h || j < 0 || j >= w) {
            return false;
        }

        char ch = mat[i * w + j];
        // 判断是否访问过
        // 如果没有访问过，判断是否和字符串str[pos]匹配
        if (ch == '#' || ch != str[pos]) {
            return false;
        }

         // 如果匹配，判断是否匹配到最后一个字符
        if (pos + 1  == str_len) {
            return true;
        }

        // 说明当前字符成功匹配，标记一下，下次不能再次进入
        mat[i * w + j] = '#';
        for (int k = 0; k < 4; ++k) {
            if (dfs(i + dir[k], j + dir[k + 1], pos + 1, str)) {
                return true;
            }
        } 
        // 如果4个方向都无法匹配 str[pos + 1]
        // 则回溯， 将'#' 还原成 ch          
        mat[i * w + j] = ch;
        // 说明此次匹配是不成功的
        return false;   
    }

    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        mat = matrix;
        h = rows, w = cols;
        str_len = strlen(str);
        //每个格子都作为开始结点一次
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (dfs(i, j, 0, str)) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

### 机器人的运动范围

最开始，我们在(0,0)的位置，我们假设按照{右，下，左，上}的方向去试探。所以我们走的顺序应该是按照图中的下标走的。
当走到4的时候，发现不能往继续往右边走，并且4个方向都走不通了，那就回溯到3,发现可以走到5，接着就站在5的视角，发现可以走6，就一直按照这个想法。

本题的递归函数就是：首先站在(0,0)的视角，先往右试探，发现可以走，就以下一个为视角，继续做相同的事情。
递归函数模板为：

dfs遍历
最开始，我们在(0,0)的位置，我们假设按照{右，下，左，上}的方向去试探。所以我们走的顺序应该是按照图中的下标走的。
当走到4的时候，发现不能往继续往右边走，并且4个方向都走不通了，那就回溯到3,发现可以走到5，接着就站在5的视角，发现可以走6，就一直按照这个想法。

```
class Solution {

public:
    using V = vector<int>;
    using VV = vector<V>;
    //方向数组，向左  向右 向前向后 
    int dir[5] = {-1, 0, 1, 0, -1};
    
    //判断函数
    int check(int n) {
        int sum = 0;

        while (n) {
            sum += (n % 10);
            n /= 10;
        }

        return sum;
    }

    void dfs(int x, int y, int sho, int r, int c, int &ret, VV &mark) {
        // 检查下标 和 是否访问
        if (x < 0 || x >= r || y < 0 || y >= c || mark[x][y] == 1) {
            return;
        }
        // 检查当前坐标是否满足条件
        if (check(x) + check(y) > sho) {
            return;
        }
        // 代码走到这里，说明当前坐标符合条件
        mark[x][y] = 1;
        ret += 1;
        //上下左右四个方向
        for (int i = 0; i < 4; ++i) {
            dfs(x + dir[i], y + dir[i + 1], sho, r, c, ret, mark);
        }



    } 
    int movingCount(int sho, int rows, int cols)
    {
        if (sho <= 0) {
            return 0;
        }

        VV mark(rows, V(cols, -1));
        int ret = 0;
        dfs(0, 0, sho, rows, cols, ret, mark);
        return ret;
    }
};
```

### 剪绳子

题目

给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1，m<=n），每段绳子的长度记为k[1],...,k[m]。请问k[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

动态规划

当 绳子长 i < = 4 最长是,属于特殊情况

当绳子长度大于3
    i < = 4 f[i] = i
    i > 4 时 f[i] = max(f[i],j * f[i-j])

```
class Solution {
public:
    int cutRope(int number) {
        if (number == 2) {
            return 1;
        }
        else if (number == 3) {
            return 2;
        }

        vector<int> f(number + 1, -1);
        for (int i = 1; i <= 4; ++i) {
            f[i] = i;
        }
        for (int i = 5; i <= number; ++i) {
            for (int j = 1; j < i; ++j) {
                f[i] = max(f[i], j * f[i - j]);
            }
        }
        return f[number];
    }
};
```

### 数据库索引

```
数据库索引，是数据库管理系统中一个排序的数据结构，以协助快速查询、更新数据库表中数据。索引的实现通常使用B树及其变种B+树。
数据库系统还维护着满足特定查找算法的数据结构，这些数据结构以某种方式引用（指向）数据，这样就可以在这些数据结构上实现高级查找算法。这种数据结构，就是索引
为表设置索引要付出代价的：一是增加了数据库的存储空间，二是在插入和修改数据时要花费较多的时间(因为索引也要随之变动)。
二叉查找树，每个节点分别包含索引键值和一个指向对应数据记录物理地址的指针，这样就可以运用二叉查找在O(log2n)的复杂度内获取到相应数据。

创建索引可以大大提高系统的性能。

第一，通过创建唯一性索引，可以保证数据库表中每一行数据的唯一性。

第二，可以大大加快数据的检索速度，这也是创建索引的最主要的原因。

第三，可以加速表和表之间的连接，特别是在实现数据的参考完整性方面特别有意义。

第四，在使用分组和排序子句进行数据检索时，同样可以显著减少查询中分组和排序的时间。

第五，通过使用索引，可以在查询的过程中，使用优化隐藏器，提高系统的性能。

缺点：
第一，创建索引和维护索引要耗费时间，这种时间随着数据量的增加而增加。

第二，索引需要占物理空间，除了数据表占数据空间之外，每一个索引还要占一定的物理空间，如果要建立聚簇索引，那么需要的空间就会更大。

第三，当对表中的数据进行增加、删除和修改的时候，索引也要动态的维护，这样就降低了数据的维护速度。

根据数据库的功能，可以在数据库设计器中创建三种索引：唯一索引、主键索引和聚集索引

唯一索引
唯一索引是不允许其中任何两行具有相同索引值的索引。当现有数据中存在重复的键值时，大多数数据库不允许将新创建的唯一索引与表一起保存。数据库还可能防止添加将在表中创建重复键值的新数据

主键索引

数据库表经常有一列或列组合，其值唯一标识表中的每一行。该列称为表的主键

在数据库关系图中为表定义主键将自动创建主键索引，主键索引是唯一索引的特定类型。该索引要求主键中的每个值都唯一。当在查询中使用主键索引时，它还允许对数据的快速访问。

聚集索引

在聚集索引中，表中行的物理顺序与键值的逻辑（索引）顺序相同。一个表只能包含一个聚集索引。

如果某索引不是聚集索引，则表中行的物理顺序与键值的逻辑顺序不匹配。与非聚集索引相比，聚集索引通常提供更快的数据访问速度。
```

### 50 去除数组中的重复数字

题目描述
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中第一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
返回描述：
如果数组中有重复的数字，函数返回true，否则返回false。
如果数组中有重复的数字，把重复的数字放到参数duplication[0]中。（ps:duplication已经初始化，可以直接赋值使用。）

1 hash

```
方法一：哈希+遍历
题目中含有重复的字眼，第一反应应该想到哈希，set。这里我们用哈希来解。
算法步骤：

开辟一个长度为n的vector<bool>, 初始化为false</bool>
遍历数组，第一次遇到的数据，对应置为true
如果再一次遇到已经置为true的数据，说明是重复的。返回即可。
代码如下：

bool duplicate(int numbers[], int length, int* duplication) {
        if(length < 2)
            return false;
        vector<bool> f(length,false);
        for(int i = 0; i < length; i++) {
            if(!f[numbers[i]])
                f[numbers[i]] = true;
            else {
                *duplication = numbers[i];
                return true;
            }
        }
        return false;
    }

```

class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int ret = 0;
        for (const int k : data) ret ^= k;
        ret &= (-ret);
        *num1 = 0, *num2 = 0;
        for (const int k : data) {
            if (k & ret) *num1 ^= k;
            else *num2 ^= k;
        }
    }
};