#include <iostream>
int main(){
    std::cout << "__cplusplus = " << __cplusplus << '\n';
#ifdef _MSVC_LANG
    std::cout << "_MSVC_LANG   = " << _MSVC_LANG << '\n';
#endif
    return 0;
}