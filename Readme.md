stats
=======
stats is a header-only statistics, probability, and combinatorics C++ library.

N.B. The documentation is forthcoming. The combinatorics and probability functions are a work in progress.

Getting Started
---------------
1.    Download and install pre-requisite libraries:
      * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
      * [gcem](https://github.com/kthohr/gcem#status-and-documentation) 
        * N.B. this is a header-only library that is included in the include directory of this very project. So if you follow the instructions, you will include this for free.

2.    Clone this repository
```commandline
git clone https://github.com/zborffs/stats.git
```
3.    Copy the contents of the "include" directory of _this_ project into _your_ project

Ex. 
![alt text](docs/res/IncludeDirectory.png "Copy-Paste!")

4.    Include the directory in the compilation

CMake:
```cmake
include_directories(include)
```
Makefile:
```makefile
g++ -Iinclude ... main.cpp -o main
```
5.    Include the headers when needed.
```c++
#include "stats/stats.hpp"

std::vector<double> my_data;
...
auto iqr = stats::interquartile_range(my_data.begin(), my_data.end();
...
```

Links
-----
| Topic                          | Link                                                          |
|--------------------------------|:--------------------------------------------------------------|
|Central Tendency                | https://en.wikipedia.org/wiki/Central_tendency                |
|Bootstrapping                   | https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29  |
|Skewness                        | https://en.wikipedia.org/wiki/Skewness                        |
|Pearson Correlation Coefficient | https://en.wikipedia.org/wiki/Pearson_correlation_coefficient |
|Glicko2                         | http://www.glicko.net/glicko/glicko2.pdf                      |