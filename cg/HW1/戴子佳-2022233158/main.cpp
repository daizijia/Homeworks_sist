/* input: n, points x,y,z
 * output: convex surface
 * Author: Dai zi jia*/

#include<iostream>
#include<cmath>
#include<algorithm>
#include<fstream>
#include<vector>
#include<map>
#include<stack>

using namespace std;

const int MAX = 1000;
int size = MAX;

struct Point{
    double x;
    double y;
    double z;
    int index;

    Point(double a = 0, double b = 0, double c = 0, int i = -1){ x = a; y = b; z = c; index = i;}
}P[MAX];

struct Surface{
    Point p0;
    Point p1;
    Point p2;

    Surface() {};
    Surface(Point a, Point b, Point c) { p0 = a; p1 = b; p2 = c; }
}S[MAX];

/*basic operation*/
Point add(const Point& l, const Point& r){
    Point result;
    result.x = l.x + r.x;
    result.y = l.y + r.y;
    result.z = l.z + r.z;
    return result;
}
Point sub(const Point& l, const Point& r){
    Point result;
    result.x = l.x - r.x;
    result.y = l.y - r.y;
    result.z = l.z - r.z;
    return result;
}
Point mul(const Point& p, double ratio)
{
    Point result;
    result.x = p.x * ratio;
    result.y = p.y * ratio;
    result.z = p.z * ratio;
    return result;
}
Point div(const Point& p, double ratio)
{
    Point result;
    result.x = p.x / ratio;
    result.y = p.y / ratio;
    result.z = p.z / ratio;
    return result;
}

/*about vec*/
double length(const Point& vec)
{
    return (sqrt(pow(vec.x, 2) + pow(vec.y, 2) + pow(vec.z, 2)));
}
Point multiply(const Point& vec1, const Point& vec2) // vec1 x vec2
{
    Point result;
    result.x = vec1.y * vec2.z - vec2.y * vec1.z;
    result.y = vec1.z * vec2.x - vec2.z * vec1.x;
    result.z = vec1.x * vec2.y - vec2.x * vec1.y;
    return result;
}
Point normalize(const Point& vec) // to 1
{
    Point res;
    res = div(vec, length(vec));
    return res;
}
double dot(const Point& vec1, const Point& vec2){
    double res;
    res = vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    return res;
}

/*about point and surface*/
Point Normal(const Surface &s){
    Point vec1 = sub(s.p1, s.p0);
    Point vec2 = sub(s.p2, s.p0);
    return normalize(multiply(vec1, vec2));
}
bool aboveSurface(const Point &p,const Surface &s){
    bool res;
    res = dot(sub(p,s.p0), Normal(s)) >= 0;
    return res;
}

/*data process*/
double decimals(){
    double eps;
    double decimal;
    eps = 1e-12;
    decimal = ((double)rand() / RAND_MAX - 0.5) * eps;
    return decimal;
}
void addDecimals(Point P[], int n){ //add random decimals
    for(int i = 0;i < n;i++){
        P[i].x = P[i].x + decimals();
        P[i].y = P[i].y + decimals();
        P[i].z = P[i].z + decimals();
    }
}
void generatePoint(Point P[], int n){ //save in array and txt
    fstream file("point.txt", ios::out);
    ofstream out;
    out.open("point.txt",ios::app);
    for(int i = 0;i < n;i++){
        P[i].x = 200 * (rand() / double(RAND_MAX));
        P[i].y = 200 * (rand() / double(RAND_MAX));
        P[i].z = 200 * (rand() / double(RAND_MAX));
        P[i].index = i;
        out << P[i].x << " "<<P[i].y << " "<<P[i].z << "\n";
    }
    out.close();
}
void ridDup(){

}

/*core code*/
void getConvex(Surface S[], int m, Point P[], int n){
    Surface Temp[MAX];
    bool flag[MAX][MAX];
    S[0].p0 = P[0], S[0].p1 = P[1], S[0].p2 = P[2];
    S[1].p0 = P[2], S[1].p1 = P[1], S[1].p2 = P[0];
    m = 2;
    for(int i = 3;i < n;i++){
        int count = 0;
        for(int j = 0;j < m;j++){
            bool ab;
            ab = aboveSurface(P[i],S[j]);
            if(!ab){
                Temp[count] = S[j];
                count++;
            }
            flag[S[j].p0.index][S[j].p1.index] = ab;
            flag[S[j].p1.index][S[j].p2.index] = ab;
            flag[S[j].p2.index][S[j].p0.index] = ab;
        }
        for(int j = 0;j < m;j++){
            if(flag[S[j].p0.index][S[j].p1.index] && ! flag[S[j].p1.index][S[j].p0.index]){
                Temp[count].p0 = S[j].p0, Temp[count].p1 = S[j].p1, Temp[count].p2 = P[i];
                count++;
            }
            if(flag[S[j].p1.index][S[j].p2.index] && ! flag[S[j].p2.index][S[j].p1.index]){
                Temp[count].p0 = S[j].p1, Temp[count].p1 = S[j].p2, Temp[count].p2 = P[i];
                count++;
            }
            if(flag[S[j].p2.index][S[j].p0.index] && ! flag[S[j].p0.index][S[j].p2.index]){
                Temp[count].p0 = S[j].p2, Temp[count].p1 = S[j].p0, Temp[count].p2 = P[i];
                count++;
            }
        }
        m = count;
        for(int j = 0;j < m;j++){
            S[j] = Temp[j];
            //printf("%i %i %i\n",S[j].p0.index,S[j].p1.index,S[j].p2.index);
        }
        //printf("%i\n",m);
    }
    size = m;
}

int main(){
    fstream file("surface.txt", ios::out);
    ofstream out;
    out.open("surface.txt",ios::app);
    int n = 0;
    cin >> n;
//    for(int i = 0;i < n;i++)
//    {
//        cin >> P[i].x >> P[i].y >> P[i].z;
//        P[i].index = i;
//    }
    generatePoint(P, n);
    addDecimals(P,n);
    getConvex(S,size,P,n);
    for(int i = 0;i<size;i++){
        printf("%i %i %i\n",S[i].p0.index,S[i].p1.index,S[i].p2.index);
        out << 3 <<" "<< S[i].p0.index << " "<<S[i].p1.index << " "<<S[i].p2.index << "\n";
    }
    out.close();
}