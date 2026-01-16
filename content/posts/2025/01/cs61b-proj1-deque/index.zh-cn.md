---
title: "「CS61B Proj1」链表和双端队列及简单应用"
subtitle: ""
description: ""
slug: 7c3eb9
date: 2025-01-29T15:25:59+08:00
lastmod: 2025-01-29T15:25:59+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['CS61B', 'Java', '链表', '双端队列', 'JUnit', 'Debug', '数据结构']
# 分类
categories: ["学习"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
collections: ["CS61B 学习记录"]
# 从主页面中去除
hiddenFromHomePage: false
# 从搜索中去除
hiddenFromSearch: false

lightgallery: false

# 否开启表格排序
table:
  sort: false
toc:
  enable: true
  auto: true
expirationReminder:
  enable: false
  # ...
code:
  copy: true
  # ...
edit:
  enable: false
  # ...
math:
  enable: true
  # ...
mapbox:
  accessToken: ""
  # ...
share:
  enable: true
  # ...
comment:
  enable: true
  # ...
library:
  css:
    # someCSS = "some.css"
    # 位于 "assets/"
    # 或者
    # someCSS = "https://cdn.example.com/some.css"
  js:
    # someJS = "some.js"
    # 位于 "assets/"
    # 或者
    # someJS = "https://cdn.example.com/some.js"
seo:
  images: []
  # ...
---

## 前言

在开始完成 proj1 之前，最好也将 lab 的进度推进到 lab3 上，lab3 相关的内容和 proj1 的 EC 相关，如果只需要完成基础部分，将 List 和 Deque 相关的内容完成即可。

> 21sp-proj1 相关文档: [Project 1: Data Structures](https://sp21.datastructur.es/materials/proj/proj1/proj1) \
> 我的实现代码: [cs61b-proj1](https://github.com/mulberror/CS61B/tree/main/proj1)、[cs61b-proj1EC](https://github.com/mulberror/CS61B/tree/main/proj1ec)

参照项目的文档，首先将完成主体的前半部分，该部分主要就是分别使用双端链表实现双端队列 `LinkedListDeque`、以及使用数组模拟实现双端队列 `ArrayDeque`，完成这两个部分后就可以在 Gradescope 上进行 Project 1: Checkpoint 的测试了。

需要分别实现 `LinkedListDeque` 和 `ArrayDeque`，这两个类的实现都是基于 `Deque` 接口的，项目文档给定了接口相关规范如下：

```java
// deque/Deque.java
public interface Deque<T> {
    void addFirst(T data);
    void addLast(T data);
    default boolean isEmpty() {
        return size() == 0;
    }
    int size();
    void printDeque();
    T removeFirst();
    T removeLast();
    T get(int index);
}
```

## LinkedListDeque: 使用双向链表实现双端队列

我使用的是带头节点和尾节点的双端队列进行实现，保证链表当中始终存在两个没有意义的节点 `head` 和 `tail`，这样可以减少很多边界条件的判断。

- 链表当中节点的定义如下：

```java
private static class Node<T> {
    T data;
    Node<T> next;
    Node<T> prev;

    Node(T data, Node<T> prev, Node<T> next) {
        this.data = data;
        this.next = next;
        this.prev = prev;
    }
}
```

- `addFirst` 和 `addLast` 方法实现就分别创建一个新节点，然后将新节点连接到 `head` 的 `next` 或者 `tail` 的 `prev` 上。

```java
public void addFirst(T data) {
    Node<T> newNode = new Node<T>(data, head, head.next);
    head.next.prev = newNode;
    head.next = newNode;
    size++;
}

public void addLast(T data) {
    Node<T> newNode = new Node<T>(data, tail.prev, tail);
    tail.prev.next = newNode;
    tail.prev = newNode;
    size++;
}
```

- `removeFirst` 和 `removeLast` 方法实现就是将 `head` 或者 `tail` 的 `next` 或者 `prev` 连接到下一个节点上，然后将当前节点的 `prev` 和 `next` 置空，并返回当前节点的数据。

```java
public T removeFirst() {
    if (isEmpty()) {
        return null;
    }
    Node<T> deleteNode = head.next;
    head.next = deleteNode.next;
    deleteNode.next.prev = head;
    size--;
    return deleteNode.data;
}

public T removeLast() {
    if (isEmpty()) {
        return null;
    }
    Node<T> deleteNode = tail.prev;
    tail.prev = deleteNode.prev;
    deleteNode.prev.next = tail;
    size--;
    return deleteNode.data;
}
```

- `get` 方法实现就是使用指针遍历的方式，找到对应的节点，然后返回节点的数据，需要注意边界问题，该部分是存在测试点的。
- `getRecursive` 方法实现需要使用一个辅助函数 `getRecursiveHelper` 进行递归遍历。

```java
public T get(int index) {
    if (index < 0 || index >= size) {
        return null;
    }
    Node<T> node = head.next;
    for (int i = 0; i < index && node != tail; i++) {
        node = node.next;
    }
    return node.data;
}

private T getRecursiveHelper(Node<T> node, int index) {
    if (index == 0) {
        return node.data;
    }
    return getRecursiveHelper(node.next, index - 1);
}

public T getRecursive(int index) {
    if (index < 0 || index >= size) {
        return null;
    }
    return getRecursiveHelper(head.next, index);
}
```

以上的几种方法进行测试正确性时需要重点关注，后面的测试部分主要就是针对该部分进行测试的。

- `printDeque` 方法实现就是遍历链表，将节点的数据打印出来。

```java
public void printDeque() {
    Node<T> current = head.next;
    while (current != tail) {
        System.out.print(current.data + " ");
        current = current.next;
    }
    System.out.println();
}
```

- `equals` 的功能需要重写判断相同的方法，该部分有点细节，首先就是 `LinkedListDeque` 和 `ArrayDeque` 是从 `Deque` 接口继承的，所以需要判断是否是 `Deque` 类型，该部分存在类型强制转换的问题；然后对于中间数据的比较也要使用 `equals` 方法进行比较，该部分存在拷贝和判断 `null` 的问题，需要注意。

```java
@Override
public boolean equals(Object o) {
    if (!(o instanceof Deque)) {
        return false;
    }
    if (size() != ((Deque<?>) o).size()) {
        return false;
    }
    for (int i = 0; i < size(); i++) { // Time Complexity high
        T left = (T) ((Deque<?>) o).get(i);
        T right = get(i);
        if (left == null && right == null) {
            continue;
        }
        if (left == null || right == null) {
            return false;
        }
        if (!(left.equals(right))) {
            return false;
        }
    }
    return true;
}
```

以上采用 `get` 方法进行遍历比较的时间复杂度较高，不过为了统一对 `Deque` 的两个实现进行比较，这里就不再进行优化了。

- 迭代器 `Iterator` 的实现就是实现 `Iterator` 接口，然后实现 `hasNext` 和 `next` 方法。

```java
@Override
public Iterator<T> iterator() {
    return new LinkedListIterator();
}

private class LinkedListIterator implements Iterator<T> {
    private Node<T> current = head.next;

    @Override
    public boolean hasNext() {
        return current != tail;
    }

    @Override
    public T next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        T item = current.data;
        current = current.next;
        return item;
    }
}
```

## ArrayDeque: 使用数组模拟实现双端队列

该部分我是采用一个循环数组进行实现的，容器具有两个属性 `capacity` 和 `size`，`capacity` 表示数组的容量，`size` 表示数组当前的大小。

在文档中提到未使用的元素个数不超过 25%，所以在插入和删除的时候需要进一步判断是否需要扩容或者缩容，那么我的策略就是在 `size` 达到 `capacity` 时进行扩容，当 `size` 小于 `capacity / 2` 时进行缩容，该实现也满足了文档中的要求。

```java
private void resize(int newCapacity) {
    T[] newArray = (T[]) new Object[newCapacity];
    for (int i = 0; i < size; i++) {
        newArray[i] = array[(head + i) % capacity];
    }
    array = newArray;
    head = 0;
    tail = size;
    capacity = newCapacity;
}

public void addFirst(T data) {
    head = (head - 1 + capacity) % capacity;
    array[head] = data;

    size++;
    if (size == capacity) {
        resize(capacity * 2);
    }
}

public void addLast(T data) {
    array[tail] = data;
    tail = (tail + 1) % capacity;

    size++;
    if (size == capacity) {
        resize(capacity * 2);
    }
}

public T removeFirst() {
    if (isEmpty()) {
        return null;
    }
    T data = array[head];
    array[head] = null;
    head = (head + 1) % capacity;

    size--;
    if (size * 2 < capacity && capacity > 8) {
        resize(capacity / 2);
    }
    return data;
}

public T removeLast() {
    if (isEmpty()) {
        return null;
    }
    tail = (tail - 1 + capacity) % capacity;
    T data = array[tail];
    array[tail] = null;

    size--;
    if (size * 2 < capacity && capacity > 8) {
        resize(capacity / 2);
    }
    return data;
}
```

其他相关的实现可以在我的代码中查看，这里就不再赘述了，和 `LinkedListDeque` 的实现类似，不过需要注意的是 `get` 方法的实现，因为是循环数组，所以可以直接使用 `(head + index) % capacity` 来获取对应的数据。

那么在完成以上部分后，就可以在 Gradescope 上进行 Project 1: Checkpoint 的测试了。

不过在完成测试后，请不要急于直接进行下一个测试 Project 1: Data Structures，下一阶段的测试一共有 640pt，里面有一些比较细节的测试点，包括代码风格等相关测试，而且 Gradescope 上测试频率是优先的，4 个小时内可以测试 2 次，所以请在测试前仔细检查代码并进行完善的测试:)。

## Project 1: Data Structures

该测试的内容就是除了 EC 部分的所有内容，包括 `MaxArrayDeque` 的实现、使用双端队列完成 `Guitar Hero` 该应用场景的实现等。

### MaxArrayDeque

该双端队列的功能需要在之前实现的基础上进行拓展，需要实现针对特定 Comparator 求解的 `max` 的方法，不过本身对效率没有非常高的要求，所以并不需要采用堆的知识，使用循环遍历的方式即可。

```java
public class MaxArrayDeque<T> extends ArrayDeque<T> {
    private final Comparator<T> comparator;

    public MaxArrayDeque(Comparator<T> c) {
        comparator = c;
    }

    public T max() {
        return getMaxHelper(comparator);
    }

    public T max(Comparator<T> c) {
        return getMaxHelper(c);
    }

    private T getMaxHelper(Comparator<T> c) {
        if (isEmpty()) {
            return null;
        }
        T maxResult = get(0);
        for (T item : this) {
            if (c.compare(item, maxResult) > 0) {
                maxResult = item;
            }
        }
        return maxResult;
    }
}
```

### Guitar Hero

~~这就是我选择这个文章头图的原因，波奇酱真可爱~~

主要实现的内容在 `GuitarString.java` 内，根据项目文档完成三个方法的实现即可，代码中还额外有一个方法 `sample` 方法需要实现，不过按照代码中注释的提示写也是比较简单的。

- GuitarString 构造函数需要往 `buffer` 中填充 `capacity` 个 0，$capacity = \frac{SR}{frequency}$
- `pluck()` 方法是将 `buffer` 中的数据随机填充为 $[-0.5, 0.5]$ 之间的随机数
- `tic()` 方法是将 `buffer` 中的数据进行计算，具体操作是将开头两个数字的平均数乘以 `DECAY` 然后放到 `buffer` 的最后一个位置，并且弹出开头的数字

```java
/* Create a guitar string of the given frequency.  */
public GuitarString(double frequency) {
    // DONE: Create a buffer with capacity = SR / frequency. You'll need to
    //       cast the result of this division operation into an int. For
    //       better accuracy, use the Math.round() function before casting.
    //       Your should initially fill your buffer array with zeros.
    long capacity = Math.round(SR / frequency);
    buffer = new ArrayDeque<>();
    for (int i = 0; i < capacity; i++) {
        buffer.addLast(0.0);
    }
}


/* Pluck the guitar string by replacing the buffer with white noise. */
public void pluck() {
    // Step 1
    // DONE: Dequeue everything in buffer, and replace with random numbers
    //       between -0.5 and 0.5. You can get such a number by using:
    //       double r = Math.random() - 0.5;
    //
    //       Make sure that your random numbers are different from each
    //       other. This does not mean that you need to check that the numbers
    //       are different from each other. It means you should repeatedly call
    //       Math.random() - 0.5 to generate new random numbers for each array index.
    for (int i = 0; i < buffer.size(); i++) {
        buffer.removeFirst();
        double r = Math.random() - 0.5;
        buffer.addLast(r);
    }
}

/* Advance the simulation one time step by performing one iteration of
    * the Karplus-Strong algorithm.
    */
public void tic() {
    // DONE: Dequeue the front sample and enqueue a new sample that is
    //       the average of the two multiplied by the DECAY factor.
    //       **Do not call StdAudio.play().**
    double frontDouble = buffer.removeFirst();
    double nextDouble = buffer.get(0);
    buffer.addLast((frontDouble + nextDouble) * 0.5 * DECAY);
}
```

完成该部分后，记得将注释中的 `TODO` 修改掉，不然在测试的时候也会出现 `Style` 的问题。

## Project 1: 额外挑战

该项目的 EC 并不是下面的 `Guitar Hero` 的内容，而是运用 `lab3` 中的知识，自行实现一个 `Autograder` 测试。

> 项目的文档：[proj1ec](https://sp21.datastructur.es/materials/proj/proj1/proj1ec)

其实这个 EC 的内容并不难，说简单点就是随机生成数据，将两个相同的操作分别在 `StudentArrayDeque` 和 `ArrayDequeSolution` 进行执行，然后用 `assertEquals` 进行比较，如果不相等就输出错误信息。

~~感觉写的有点丑~~

```java
public class TestArrayDequeEC {
    private static final int MAX_NUMBER = 10000;
    @Test
    public void test() {
        ArrayDequeSolution<Integer> solutionDeque = new ArrayDequeSolution<>();
        StudentArrayDeque<Integer> studentDeque = new StudentArrayDeque<>();
        String message = "";
        boolean flag = true;
        int size = 0;

        while (flag) {
            int option = StdRandom.uniform(0, 6);
            if (size == 0) {
                option = StdRandom.uniform(0, 3);
            }
            Integer number = StdRandom.uniform(MAX_NUMBER);
            switch (option) {
                case 0:
                    solutionDeque.addFirst(number);
                    studentDeque.addFirst(number);
                    size++;
                    message += "addFirst(" + number + ")\n";
                    break;
                case 1:
                    solutionDeque.addLast(number);
                    studentDeque.addLast(number);
                    size++;
                    message += "addLast(" + number + ")\n";
                    break;
                case 2:
                    if (solutionDeque.size() != studentDeque.size()) {
                        flag = false;
                    }
                    break;
                case 3:
                    int position = StdRandom.uniform(0, studentDeque.size());
                    Integer getNumber0 = solutionDeque.get(position);
                    Integer getNumber1 = studentDeque.get(position);
                    if (!getNumber0.equals(getNumber1)) {
                        flag = false;
                    }
                    break;
                case 4:
                    Integer removeFirstNumber0 = solutionDeque.removeFirst();
                    Integer removeFirstNumber1 = studentDeque.removeFirst();
                    message += "removeFirst()\n";
                    if (!removeFirstNumber0.equals(removeFirstNumber1)) {
                        assertEquals(message, removeFirstNumber0, removeFirstNumber1);
                        flag = false;
                    }
                    size--;
                    break;
                case 5:
                    Integer removeLastNumber0 = solutionDeque.removeLast();
                    Integer removeLastNumber1 = studentDeque.removeLast();
                    message += "removeLast()\n";
                    if (!removeLastNumber0.equals(removeLastNumber1)) {
                        assertEquals(message, removeLastNumber0, removeLastNumber1);
                        flag = false;
                    }
                    size--;
                    break;
            }
        }
    }
}
```