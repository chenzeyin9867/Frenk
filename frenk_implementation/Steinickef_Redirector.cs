using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Redirection;


public class Steinickef_Redirector : Redirector
{
    // User Experience Improvement Parameters
    private const float MOVEMENT_THRESHOLD = 0.2f; // meters per second
    private const float ROTATION_THRESHOLD = 1.5f; // degrees per second

    //
    private float rotationFromCurvatureGain; //Proposed curvature gain based on user speed
    private float rotationFromRotationGain; //Proposed rotation gain based on head's yaw
    private Vector2 middlePos;// 表示目前确定了目标点的位置和方向以后，路径的分割点（分割线，圆弧的一些特殊点）
    private Vector2 targetPos;//表示目前确定的目标点的物理位置
    private Vector2 targetDir;//表示目标点的物理方向
    private bool isRedirecting = false; //正在重定向过程中
    private float r,r1, r2;//圆弧的半径参数，其中r针对路径中只有一段圆弧情况，r1,r2针对路径中有两段圆弧情况
    private enum PathType  { a,b,c,d,e};
    private PathType pathType;
    private bool passMiddlePoint = false;

    public void findTarget()
    {
        isRedirecting = false;
        // .............这里待解决
        //根据论文，如果当前的方向能够与虚拟物品重叠，则找到了target就是这个虚拟物品
        //论文里说是从虚拟物品和物理物品的注册关系得到物理的targetPos跟targetDir这两个属性
        //所以这里还需要做的就是确定targetPos跟targetDir这两个属性
        //暂时还没考虑路径中存在障碍物的情况，也就是论文中的4.3内容
        //..............
        isRedirecting = true;
        calculatePath();
    }

    public void calculatePath()
    {
        Vector2 currentDir = Utilities.FlattenedDir2D(redirectionManager.currDirReal).normalized;
        Vector2 currentPos = Utilities.FlattenedPos2D(redirectionManager.currPosReal);
        float x1 = currentPos.x, y1 = currentPos.y;//(x1, y1) + m*(s1,t1) = (x2,y2)+n*(s2,t2)
        float x2 = targetPos.x, y2 = targetPos.y;
        float s1 = currentDir.x, t1 = currentDir.y;//(s1,t1)是当前方向单位向量
        float s2 = targetDir.x, t2 = targetDir.y;
        float D = s1 * (-t2) + s2 * t1;  //用cramer法则，用行列式算m,n
        float D1 = (x2 - x1) * (-t2) + s2 * (y2 - y1);
        float D2 = s1 * (y2 - y1) - t1 * (x2 - x1);
        float m = D1 / D;
        float n = D2 / D;
        
       

        if (Vector2.Dot(currentDir, targetDir) > 0 && m > 0 && n < 0)//a,b情况
        {
             Vector2 targetOrthoDir = new Vector2(-targetDir.y, targetDir.x).normalized;//目标点的法向量，方向可能有两个
            targetOrthoDir = Mathf.Sign(Vector2.Dot(targetOrthoDir, currentDir)) * targetOrthoDir;//确定目标点的法向量方向
            Vector2 currentOrthoDir = new Vector2(-currentDir.y, currentDir.x).normalized;//当前方向的法向量
            currentOrthoDir = Mathf.Sign(Vector2.Dot(currentOrthoDir, targetDir)) * currentOrthoDir;//法向量有两个，选对的那个
            float u1 = currentOrthoDir.x, v1 = currentOrthoDir.y;//(u1,v1)为(s1,t1)法向量
            float u2 = targetOrthoDir.x, v2 = targetOrthoDir.y;//(u2,v2)为(s2,t2)法向量
            //(x2-x1-u1*r,y2-y1-v1*r)·(u2,v2) = r
            r = ((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2);//求出半径
            Vector2 tangentPos = currentPos + (currentOrthoDir * (float)r) + r * targetOrthoDir;//为a中大圆弧切点的位置
            if ((targetPos - tangentPos).x / targetDir.x > 0)//a情况
            {
                pathType = PathType.a;
                middlePos = tangentPos;
            }
            else//b情况
            {
                pathType = PathType.b;
                //(x1,y1)+p(s1,t1)+r(u1,v1)+r(u2,v2)=(x2,y2)可解出p,r
                float d = s1 * (v1 + v2) - t1 * (u1 + u2);//利用cramer法则，算行列式
                float d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2);
                float d2 = s1 * (y2 - y1) - t1 * (x2 - x1);
                float p = d1 / d;
                r = d2 / d;
                middlePos = currentPos + p * currentDir;
            }
        } else if (m<0 && n < 0) //c情况
        {
            //要解非线性方程组才能算出r1,r2，这也太难了！！
            //两条圆弧的两个半径非常难求解，我想不出办法了
        }
        else//d情况
        {
            pathType = PathType.d;
            Vector2 targetOrthoDir = new Vector2(-targetDir.y, targetDir.x).normalized;//目标点的法单位向量，方向可能有两个
            targetOrthoDir = -Mathf.Sign(Vector2.Dot(targetOrthoDir, currentDir)) * targetOrthoDir;//确定目标点的法向量方向
            Vector2 currentOrthoDir = new Vector2(-currentDir.y, currentDir.x).normalized;//当前方向的法向量
            currentOrthoDir = -Mathf.Sign(Vector2.Dot(currentOrthoDir, targetDir)) * currentOrthoDir;//法向量有两个，选对的那个
            float u1 = currentOrthoDir.x, v1 = currentOrthoDir.y;//(u1,v1)为(s1,t1)法向量
            float u2 = targetOrthoDir.x, v2 = targetOrthoDir.y;//(u2,v2)为(s2,t2)法向量
            //(x1,y1)+p(s1,t1)+r(u1,v1)+r(u2,v2)=(x2,y2)可解出p,r
            float d = s1 * (v1 + v2) - t1 * (u1 + u2);//利用cramer法则，算行列式
            float d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2);
            float d2 = s1 * (y2 - y1) - t1 * (x2 - x1);
            float p = d1 / d;
            r = d2 / d;
            middlePos = currentPos + p * currentDir;
        }
        //不考虑e情况，因为可以把e情况当a情况考虑，虽然可能会有圆弧的曲率过大，但是这篇论文算法本身就不能保证所有路径曲率在max范围内


    }
    public override void ApplyRedirection()
    {
        if (isRedirecting == false || Utilities.Approximately(redirectionManager.currPosReal,targetPos))
            findTarget();
        if (isRedirecting == true)
        {
           if (Utilities.Approximately(Utilities.FlattenedPos2D(redirectionManager.currPosReal) , middlePos))
            {
                passMiddlePoint = true;
            }
            // Get Required Data
            Vector3 deltaPos = redirectionManager.deltaPos;
            float deltaDir = redirectionManager.deltaDir;
            switch (pathType)//判断路径类型
            {
                case PathType.a:
                {
                        if (!passMiddlePoint)
                        {
                            if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) //User is moving
                            {
                                //走过的角度
                                rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                            }
                        }
                        break;
                }
                case PathType.b:
                {
                    if (passMiddlePoint)
                        {
                            if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) //User is moving
                            {
                                //走过的角度
                                rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                            }
                        }
                    break;
                }
                case PathType.c:
                {
                    /////c情况如上所说，两条圆弧半径非常难算，待解决此问题
                    ///
                    break;
                }
                case PathType.d:
                {
                    if (passMiddlePoint)
                    {
                        if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) //User is moving
                        {
                            //走过的角度
                            rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                        }
                    }
                    break;
                }
            }
        }
        InjectCurvature(rotationFromCurvatureGain);
    }
}

