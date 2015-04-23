float Fibonacci(float n);

kernel void calc(global const float *a, global const float *b, global float *answer) {
  unsigned int xid = get_global_id(0);
  answer[xid] = Fibonacci(a[xid]);
}

float Fibonacci(float n) {
    if(n <= 0) return 0;
    if(n > 0 && n < 3) return 1;

    float result = 0;
    float preOldResult = 1;
    float oldResult = 1;

    for (int i=2;i<n;i++) {
        result = preOldResult + oldResult;
        preOldResult = oldResult;
        oldResult = result;
    }

    return result;
}