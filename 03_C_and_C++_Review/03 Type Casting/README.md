a. `static_cast<new_type>(expression)`


Example (`CTRL + Shift + V` to render `.md` files):
```cpp
double pi = 3.14159;
int rounded_pi = static_cast<int>(pi);
```

Explanation:
static_cast is used for implicit conversions between types. It's the most common cast and is checked at compile-time. In this example, it converts a double to an int, truncating the decimal part.


b. `dynamic_cast<new_type>(expression)`


Example:
```cpp
class Base { virtual void foo() {} };
class Derived : public Base { };

Base* base_ptr = new Derived; // upcasting. This is called upcasting, where a Derived object is treated as a Base object.
Derived* derived_ptr = dynamic_cast<Derived*>(base_ptr); // downcasting. derived_ptr will point to the same Derived object as base_ptr
```

Explanation:
dynamic_cast is used for safe downcasting in inheritance hierarchies. It performs a runtime check and returns nullptr if the cast is not possible. It requires at least one virtual function in the base class.


c. `const_cast<new_type>(expression)`


Example:
```cpp
const int constant = 10;
int* non_const_ptr = const_cast<int*>(&constant);
*non_const_ptr = 20; // Modifies the const variable (undefined behavior)
```

Explanation:
const_cast is used to add or remove const (or volatile) qualifiers from a variable. It's the only C++ style cast that can do this. However, modifying a const object leads to undefined behavior. Important: You should never use const_cast to modify variables that were originally declared as const. Doing so leads to undefined behavior.


d. `reinterpret_cast<new_type>(expression)`


Example:
```cpp
int num = 42;
char* char_ptr = reinterpret_cast<char*>(&num);
```

Explanation:
reinterpret_cast is the most dangerous cast. It can convert between unrelated types, like pointers to integers or vice versa. It's often used for low-level operations and should be used with extreme caution. The expression &num takes the address of the integer variable num. This returns a pointer of type int* reinterpret_cast<char*>(...) converts this int* pointer (pointing to an int) into a char* pointer (pointing to a char). reinterpret_cast is a type of cast in C++ that allows you to reinterpret the bits of one type as another type without any actual type-checking or conversion. It is primarily used for low-level memory manipulation. The memory location of num is reinterpreted as a char*. This allows you to access the individual bytes of the integer in memory. Since num is a 4-byte integer, char_ptr now points to the first byte of that 4-byte memory block. You can access and manipulate the memory byte-by-byte through this pointer. For example, if you were to dereference char_ptr or access successive bytes like char_ptr[0], char_ptr[1], char_ptr[2], char_ptr[3], you would see the raw bytes of the integer 42 in memory.

