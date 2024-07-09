---
title: Go语言学习-函数
categories: 技术研究
date: 2024-07-09 16:41:53
tags: [Golang]
cover:
top_img:
---

# 函数

`Go`语言中，函数的命名定义需要`func`来作为唯一标识，并且使用首字母大小写来区分，在当前文件夹下的某一个函数是否可以通过`import bag`的方式来对其他文件的可见性，如果是大写则说明可以导入，小写则只能在当前文件内可见。

函数通过四个属性来唯一确定函数签名-函数名、形参列表、返回值列表、函数体。

```go
func name(parameter-list) (result-list) {
    body
}
```

*多返回值*

在`Go`中，一个函数可以返回多个值，并且函数的返回值必须要有变量来接收，如果我们不需要某一个返回值，通常我们会用`_`下划线来接收这个返回值，作为接收某一个返回值的占位符。

我们通常想要保留函数运行过程中的某一些局部变量的结果，或者想要拥有多个返回变量，比较常见的方法就是，定义一个全局变量，并把变量作为引用类型传入到函数内，这样的方式可以达到效果，但是会有参数列表冗余的现象，如果我们需要保留的局部变量的参数非常多，那么也需要定义多个参数来一一完成。

使用多返回值可以更清晰的表达结果，避免全局变量定义的冗余，以及引用传入的冗余，我们可以将局部变量返回，并在全局中定义接收。

使用多返回值的另一个好处就是错误的处理更加方便，通常我们会将错误作为函数的最后一个返回值。这允许调用者很容易地判断操作是否成功，而不必单独检查错误变量或异常。

```go
func divide(dividend, divisor float64) (float64, error) {
    if divisor == 0.0 {
        return 0.0, errors.New("cannot divide by zero")
    }
    return dividend / divisor, nil
}
```

## 错误

## 函数值

在`Go`中，函数被看作第一类值（第一类值意思就是说明这一个值可以像基本数据类型一样使用），具体来说：被赋值给变量、作为参数传递给函数、作为函数的返回值、在运行时动态创建、被存储在数据结构中。

函数类型的零值是`nil`，调用值为`nil`的函数会引起`panic`错误，而且函数可以与`nil`进行比较。但是函数与函数之间时不可以比较的，也不能使用函数值作为`map`的`key`。

```go
func square(n int) int { return n * n }
func negative(n int) int { return -n }
func product(m, n int) int { return m * n }

f := square
fmt.Println(f(3)) // "9"

f = negative
fmt.Println(f(3))     // "-3"
fmt.Printf("%T\n", f) // "func(int) int"

f = product // compile error: can't assign func(int, int) int to func(int) int
```

函数值使得我们不仅仅可以通过数据来参数化函数，也可以通过行为。

> `strings.Map()`是一个高阶函数，它允许你对字符串中的每个字符执行一个指定的映射操作。这个函数接受两个参数：第一个参数是一个映射函数，此映射函数会被应用到字符串中的每个字符上；第二个参数是要进行操作的字符串。映射函数需要接收一个rune类型的值，并返回一个rune类型的值。如果映射函数返回负值，则该字符会从结果字符串中被删除。

```go
func add1(r rune) rune { return r + 1 }

fmt.Println(strings.Map(add1, "HAL-9000")) // "IBM.:111"
fmt.Println(strings.Map(add1, "VMS"))      // "WNT"
fmt.Println(strings.Map(add1, "Admix"))    // "Benjy"
```

## 匿名函数

## 可变参数

## Defferred函数

## Panic异常

## Recover捕获异常
