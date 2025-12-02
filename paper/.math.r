]\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=4[/math]

In Alon Amit's answer, he mentions that the equation is homogeneous, since if [math](a,b,c)[/math] is a solution, so is [math](ka,kb,kc)[/math], because substituting [math](ka,kb,kc)[/math] into the equation causes all the "[math]k[/math]"s to cancel out. We use that to our advantage here. Without loss of generality, we can assume that [math]c=1[/math], and look for rational solutions to [math](a,b)[/math], since we can then multiply by a common denominator to find integer solutions. Let's start.

[math]\frac{a}{b+1}+\frac{b}{a+1}+\frac{1}{a+b}=4[/math]

Multiply out the denominators:

[math]a(a+1)(a+b)+b(b+1)(a+b)+(a+1)(b+1)=4(a+1)(b+1)(a+b)[/math]

Distribute, rearrange, and combine like terms, but do not cancel across the equal sign.

[math]a^3+a^2b+ab^2+b^3+a^2+2ab+b^2+ab+a+b+1=4a^2b+4ab^2+4a^2+8ab+4b^2+4a+4b[/math]

See the first four terms of the left side? We can then add [math]2a^2b+2ab^2[/math] to both sides, making the first four terms a perfect cube.

[math]a^3+3a^2b+3ab^2+b^3+a^2+2ab+b^2+ab+a+b+1=6a^2b+6ab^2+4a^2+8ab+4b^2+4a+4b[/math]

Then:

[math](a+b)^3+(a+b)^2+(a+b)+1+ab=6ab(a+b)+4(a+b)^2+4(a+b)[/math]

Look at the first four terms of the left side. Again, we add [math]2(a+b)^2+2(a+b)[/math] to both sides to make these four terms a perfect cube.

[math](a+b)^3+3(a+b)^2+3(a+b)+1+ab=6ab(a+b)+6(a+b)^2+6(a+b)[/math]
Group things together:

[math](a+b+1)^3+ab=6(a+b)(ab+a+b+1)[/math]

Now, designate [math]x=a+b+1[/math] and [math]z=ab[/math]. Therefore, [math]a,b=\frac{x-1\pm\sqrt{(x-1)^2-4z}}{2}[/math]. For [math]a,b[/math] to be rational, [math](x-1)^2-4z[/math] must be a rational square. Now, rewrite the last equation in [math]a,b[/math] with [math]x,z[/math]:

[math]x^3+z=6(x-1)(x+z)[/math]

In terms of [math]z[/math]:

[math]z=\frac{x^3-6x^2+6x}{6x-7}[/math]

From earlier, we know that for [math]a,b[/math] to be rational, [math](x-1)^2-4z[/math] must be a rational square. Substituting [math]z=\frac{x^3-6x^2+6x}{6x-7}[/math], we then find out that [math]\frac{2x^3+5x^2-4x-7}{6x-7}[/math] must be a rational square. We then multiply the numerator and denominator by [math]6x-7[/math] to make the denominator square. The resulting fraction is [math]\frac{12x^4+16x^3-59x^2-14x+49}{(6x-7)^2}[/math]. Since the denominator is square, we only need to worry about the numerator, assigning it a new variable: [math]y^2=12x^4+16x^3-59x^2-14x+49[/math]. We now have the following:

[math](x-1)^2-4z=\frac{y^2}{(6x-7)^2}[/math]

Substituting back into our formula for [math](a,b)[/math] from [math](x,z)[/math], we then get this:

[math]a,b=\frac{6x^2-13x+7\pm y}{2(6x-7)}[/math]

Rational solutions for [math](x,y)[/math] correspond to rational solutions for [math](a,b)[/math], from which integer solutions can be found for [math](a,b,c)[/math]. Here is the curve again, which I will call a quartic curve, since it looks like a Weierstrauss elliptic curve, but with a quartic polynomial on the right-hand side instead of a cubic one.

[math]y^2=12x^4+16x^3-59x^2-14x+49[/math]

Inspecting the curve, I was able to see four rational solutions (up to negation), though you may be able to see more:

[math](x,y)=(-3,10),(-1,0),(0,7),(1,2)[/math]

In order, I will label these points [math]A[/math], [math]B[/math], [math]C[/math], and [math]D[/math]. Now, what we need is a point addition method, like on an elliptic curve. And it turns out, that one exists. On an elliptic curve, you can connect two rational points with a line, and it'll intersect the elliptic curve at a third rational point. And it turns out that on a quartic curve, that you can connect three rational points with a parabola, and it'll intersect the quartic curve at a fourth rational point. There's a reason too. Suppose the parabola is [math]y=Ax^2+Bx+C[/math] ([math]A[/math], [math]B[/math], and [math]C[/math] are coefficients here, not points). Then we have [math]y=(Ax^2+Bx+C)^2=12x^4+16x^3-59x^2-14x+49[/math]. This reduces to a quartic polynomial with rational coefficients, but we already know three of the roots ([math]A[/math], [math]B[/math], and [math]C[/math] are always rational, and [math]A^2\neq 12[/math] since [math]A[/math] is rational), since they are simply the x-coordinates of the three points that we connected with that parabola. If we divide out those roots, we get a linear polynomial with rational coefficients. That fourth root (by which I mean root #4, not raising something to the power of [math]\frac{1}{4}[/math]) is the x-coordinate of the new rational point, after which we can plug it into the parabola to obtain the y-coordinate of the new rational point. To obtain [math]A[/math], [math]B[/math], and [math]C[/math], we can use a system of linear equation for our points [math](x_0,y_0),(x_1,y_1),(x_2,y_2)[/math].

[math]x_0^2 A+x_0 B+C=y_0[/math]

[math]x_1^2 A+x_1 B+C=y_1[/math]

[math]x_2^2 A+x_2 B+C=y_2[/math]

Given three rational points, we can solve these equations for [math]A[/math], [math]B[/math], and [math]C[/math], construct a connecting parabola, and find its fourth intersection with the quartic curve to get a new rational point. I will call this the [math]P[/math] operation. In other words, given three points [math]p_0[/math], [math]p_1[/math], and [math]p_2[/math], the rational point obtained by adding them is [math]P(p_0,p_1,p_2)[/math]. I will also define the negation of a point as follows: If [math]p=(x,y)[/math], then [math]-P=(x,-y)[/math]. Negation is useful, since without it, we couldn't find any new points past the fourth one. Adding the fourth point to two of the original three points would give back the third, which wouldn't be very useful. Negation allows us to find new points for possibly forever. So, let's go back to our original four points. Using these, we can get six more, up to negation:

[math]E=P(A,B,C)=(-14,637)[/math]

[math]F=P(B,C,D)=(\frac{7}{6},0)[/math]

[math]G=P(A,C,D)=(\frac{14}{11},\frac{35}{121})[/math]

[math]H=P(-A,B,D)=(\frac{15}{11},\frac{182}{121})[/math]

[math]I=P(A,B,D)=(\frac{5}{3},\frac{16}{3})[/math]

[math]J=P(-A,B,C)=(\frac{7}{2},42)[/math]

In my original exploration, I tried adding all possible triplets of points to get new points. I noticed that these new points seem to come in sets of six, where the denominators have relatively similar sizes. Interesting, each point in each set can be obtained by adding one point in the previous set, and two of the original four points, though starting from the third set, only [/math]A[/math] and [/math]B[/math] seem to be necessary. The second set is as follows:

[math]K=P(-A,B,E),L=P(-A,C,E),M=P(-A,B,G),N=P(A,B,H),O=P(-A,B,I),P=P(A,B,J)[/math]

The third set:

[math]Q_0=P(A,B,K),R_0=P(A,B,L),S_0=P(A,B,M),T_0=P(-A,B,M),U_0=P(A,B,N),V_0=P(-A,B,O)[/math]

The fourth set follows from the third set in the same way, except that the signs of the "[math]A[/math]"s are flipped. For example, [math]Q_1=P(-A,B,Q_0)[/math]. In the fifth set, the signs flip back. This flipping continues to infinity. For each point, we compute [math](a,b)[/math], and they're not all positive. And then we reach [math]V_6[/math]. This time, we compute [math](a,b)[/math], and they're both positive. So we compute [math](a,b,c)[/math], and get this out.

[math]a=4373612677928697257861252602371390152816537558161613618621437993378423467772036[/math]

[math]b=36875131794129999827197811565225474825492979968971970996283137471637224634055579[/math]

[math]c=154476802108746166441951315019919837485664325669565431700026634898253202035277999[/math]

Somewhere farther down the point series, I got another solution. I can't tell which point it is, since the online Wolfram computable notebook service (at https://develop.open.wolfram.com/app/view/newNotebook?ext=nb (http://develop.open.wolfram.com/app/view/newNotebook?ext=nb) but now gives a 404) seems to have been taken down. I do, however, still have that solution.

[math]a=32343421153825592353880655285224263330451946573450847101645239147091638517651250940206853612606768544181415355352136077327300271806129063833025389772729796460799697289[/math]

[math]b=16666476865438449865846131095313531540647604679654766832109616387367203990642764342248100534807579493874453954854925352739900051220936419971671875594417036870073291371[/math]

[math]c=18438651467072329521991466669103809627503176533640434051668643025780389550623758060258285903998125757038016122166239815379429082156904518238560341886750920963276835983515[/math]

I found even more solutions in a later investigation, but due to size I won't post them here. If I can figure out a way to find which points they corresponded to, I'll happily give out the names of the points. Since it will likely use a computer algebra package, I will also share the code for point addition and solution recovery if I can rewrite said code for said package.

EDIT: As promised, here is the code. I used SageMathCell for this.

def k(p,q,r): 
    return (r[0]-q[0])*(q[0]-p[0])*(p[0]-r[0]) 
def a(p,q,r):#Coefficients, not the actual values for the (a,b,c) triples 
    return (p[1]*(q[0]-r[0])+q[1]*(r[0]-p[0])+r[1]*(p[0]-q[0]))/k(p,q,r) 
def b(p,q,r): 
    return (p[1]*(r[0]^2-q[0]^2)+q[1]*(p[0]^2-r[0]^2)+r[1]*(q[0]^2-p[0]^2))/k(p,q,r) 
def c(p,q,r): 
    return (p[1]*q[0]*r[0]*(q[0]-r[0])+q[1]*r[0]*p[0]*(r[0]-p[0])+r[1]*p[0]*q[0]*(p[0]-q[0]))/k(p,q,r) 
def xsum(p,q,r): 
    return p[0]+q[0]+r[0] 
def px(p,q,r): 
    return (-2*a(p,q,r)*b(p,q,r)-a(p,q,r)^2*xsum(p,q,r)+12*xsum(p,q,r)+16)/(-12+a(p,q,r)^2) 
def py(p,q,r): 
    return a(p,q,r)*px(p,q,r)^2+b(p,q,r)*px(p,q,r)+c(p,q,r) 
def p(p,q,r): 
    return [px(p,q,r),py(p,q,r)] 
def n(p): 
    return [p[0],-p[1]] 
def ab(p): 
    l=6*p[0]^2-13*p[0]+7 
    r=2*(6*p[0]-7) 
    return [(l+p[1])/r,(l-p[1])/r] 
def abc(p): 
    mapped=ab(p) 
    multiplier=lcm(denominator(mapped[0]),denominator(mapped[1])) 
    return [mapped[0]*multiplier,mapped[1]*multiplier,multiplier] 
def find_point(name,index=0): 
    A=[-3,10] 
    B=[-1,0] 
    C=[0,7] 
    D=[1,2] 
    if name=="A": 
        return A 
    elif name=="B": 
        return B 
    elif name=="C": 
        return C 
    elif name=="D": 
        return D 
    elif name=="E": 
        return p(A,B,C) 
    elif name=="F": 
        return p(B,C,D) 
    elif name=="G": 
        return p(A,C,D) 
    elif name=="H": 
        return p(n(A),B,D) 
    elif name=="I": 
        return p(A,B,D) 
    elif name=="J": 
        return p(n(A),B,C) 
    elif name=="K": 
        return p(n(A),B,find_point("E")) 
    elif name=="L": 
        return p(n(A),C,find_point("E")) 
    elif name=="M": 
        return p(n(A),B,find_point("G")) 
    elif name=="N": 
        return p(A,B,find_point("H")) 
    elif name=="O": 
        return p(n(A),B,find_point("I")) 
    elif name=="P": 
        return p(A,B,find_point("J")) 
    elif name=="Q" and index==0: 
        return p(A,B,find_point("K")) 
    elif name=="R" and index==0: 
        return p(A,B,find_point("L")) 
    elif name=="S" and index==0: 
        return p(A,B,find_point("M")) 
    elif name=="T" and index==0: 
        return p(n(A),B,find_point("N")) 
    elif name=="U" and index==0: 
        return p(A,B,find_point("O")) 
    elif name=="V" and index==0: 
        return p(n(A),B,find_point("P")) 
    elif ((name=="Q" or name=="R" or name=="S" or name=="U") and index%2==0) or ((name=="T" or name=="V") and index%2==1): 
        return p(A,B,find_point(name,index-1)) 
    elif ((name=="T" or name=="V") and index%2==0) or ((name=="Q" or name=="R" or name=="S" or name=="U") and index%2==1): 
        return p(n(A),B,find_point(name,index-1)) 
Point [math]V_6[/math] could be found by using find_point("V",6).

EDIT 2: The second solution I mentioned corresponds to point [math]R_{10}[/math].

Edit 3: The third solution I found corresponds to point [math]Q_{14}[/math].

Edit 4: The fourth solution I found corresponds to point [math]U_{18}[/math].

Edit 5: The seventh solution I found corresponds to point [math]R_{36}[/math].
