int Euklid(int a, int b);

kernel void calc(global const float *a, global const float *b, global float *answer) {
  unsigned int xid = get_global_id(0);
  answer[xid] = Euklid(a[xid],b[xid]);
}

int Euklid(int a, int b)
{
    if (a == 0)                          /**Wenn a=0 ist b der groesste gemeinsame Teiler laut Definition**/
    {
    	return b;
    }
    while(b != 0)                        /**So lange wiederholen, wie b nicht 0 ist.**/
    {
    	if (a > b)
    	{
    		a = a - b;               /**Wenn a grosster als b, subtrahiere b von a.**/
    	}
        else
    	{
    		b = b - a;               /**In jedem anderen Fall subtrahiere a von b.**/
    	}
    }
    return a;                            /**In a steht jetzt der groesste gemeinsame Teiler von a und b.**/
}