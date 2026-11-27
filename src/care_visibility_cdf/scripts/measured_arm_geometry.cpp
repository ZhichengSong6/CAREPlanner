// Numeric evaluation of the calibrated arm primitives used by the measured goal and camera proposals.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using V=Eigen::Vector3d;
using M=Eigen::Matrix3d;
using RowM=Eigen::Matrix<double,3,3,Eigen::RowMajor>;
using Mat4=Eigen::Matrix<double,4,4,Eigen::RowMajor>;
struct Pose { M rotation; V center; };
struct Frame { int parent,joint; Pose origin; M outer,skew; };
struct Solid { int frame,kind; Pose origin; V half; };
struct Arm {
  std::vector<Frame> frames;
  std::vector<Solid> solids;
  V lower;
  Eigen::Vector3i shape;
  double pitch;
};
struct Node {
  V lo,hi;
  int begin,end,left,right;
};
struct Cloud {
  std::vector<V> points;
  std::vector<int> order;
  std::vector<Node> nodes;
  int split(int begin,int end) {
    Node node{V::Constant(std::numeric_limits<double>::infinity()),
              V::Constant(-std::numeric_limits<double>::infinity()),begin,end,-1,-1};
    for (int j=begin;j<end;++j) {
      node.lo=node.lo.cwiseMin(points[order[j]]);
      node.hi=node.hi.cwiseMax(points[order[j]]);
    }
    const int index=nodes.size();
    nodes.push_back(node);
    if (end-begin>16) {
      Eigen::Index axis; (node.hi-node.lo).maxCoeff(&axis);
      const int mid=(begin+end)/2;
      std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,
                       [&](int i,int j){return points[i][axis]<points[j][axis];});
      const int left=split(begin,mid),right=split(mid,end);
      nodes[index].left=left; nodes[index].right=right;
    }
    return index;
  }
};

static Pose matrix_pose(const double* a) {
  Eigen::Map<const Mat4> m(a);
  return {m.topLeftCorner<3,3>(),m.topRightCorner<3,1>()};
}
static Pose compose(const Pose& a,const Pose& b) {
  return {a.rotation*b.rotation,a.center+a.rotation*b.center};
}
static void place(const Arm& a,const double* q,std::vector<Pose>& frames,std::vector<Pose>& solids) {
  frames[0]={M::Identity(),V::Zero()};
  for (size_t i=1;i<a.frames.size();++i) {
    const Frame& f=a.frames[i];
    Pose next=compose(frames[f.parent],f.origin);
    if (f.joint>=0) {
      const double c=std::cos(q[f.joint]),s=std::sin(q[f.joint]);
      next.rotation=next.rotation*(c*M::Identity()+(1.-c)*f.outer+s*f.skew);
    }
    frames[i]=next;
  }
  for (size_t i=0;i<solids.size();++i)
    solids[i]=compose(frames[a.solids[i].frame],a.solids[i].origin);
}
static double signed_box(const V& local,const V& half) {
  const V d=local.cwiseAbs()-half;
  return d.cwiseMax(0.).norm()+std::min(0.,d.maxCoeff());
}
static bool inside(const Solid& solid,const V& local) {
  constexpr double eps=1.e-8;
  if (solid.kind==0) return ((local.cwiseAbs()-solid.half).cwiseMax(0.)).squaredNorm()<=eps*eps;
  if (solid.kind==2) return local.squaredNorm()<=(solid.half[0]+eps)*(solid.half[0]+eps);
  const double r=std::max(0.,std::hypot(local[0],local[1])-solid.half[0]);
  const double h=std::max(0.,std::abs(local[2])-solid.half[2]);
  return r*r+h*h<=eps*eps;
}
extern "C" {
void* measured_cloud_create(const double* data,int count) {
  auto* result=new Cloud;
  result->points.reserve(count);result->order.resize(count);
  for (int i=0;i<count;++i) result->points.push_back(Eigen::Map<const V>(data+3*i));
  std::iota(result->order.begin(),result->order.end(),0);
  if (count) result->split(0,count);
  return result;
}
void measured_cloud_destroy(Cloud* c) { delete c; }
// Signed clearance to centers in the received map, for deforming a proposed
// route. Each column is the minimum over the full articulated body.
static double solid_distance(const Solid& s,const V& local) {
  if (s.kind==0) return signed_box(local,s.half);
  if (s.kind==2) return local.norm()-s.half.x();
  const double axial=std::abs(local.z())-s.half.z(),radial=local.head<2>().norm()-s.half.x();
  return std::hypot(std::max(axial,0.),std::max(radial,0.))+std::min(0.,std::max(axial,radial));
}
void measured_arm_map_clearance(const Arm* arm,const double* qs,int n,const uint8_t* labels,
                                double radius,double* output,double* derivative) {
  const Arm& a=*arm;
  std::vector<Pose> frames(a.frames.size()),poses(a.solids.size());
  for (int j=0;j<n;++j) {
    place(a,qs+7*j,frames,poses);
    double* row=output+2*j;
    row[0]=radius;row[1]=radius;
    std::array<int,2> piece{{-1,-1}};
    std::array<V,2> points;
    for (size_t k=0;k<poses.size();++k) {
      const Solid& s=a.solids[k];
      const Pose& p=poses[k];
      const V bound=p.rotation.cwiseAbs()*s.half+V::Constant(radius);
      const V relative=p.center-a.lower;
      const auto lo=((relative-bound)/a.pitch).array().ceil().cast<int>().matrix().cwiseMax(0).eval();
      const auto hi=((relative+bound)/a.pitch).array().floor().cast<int>().matrix().cwiseMin(a.shape-Eigen::Vector3i::Ones()).eval();
      const M inverse=p.rotation.transpose();
      for (int x=lo[0];x<=hi[0];++x) for (int y=lo[1];y<=hi[1];++y)
        for (int z=lo[2];z<=hi[2];++z) {
          const int type=labels[(x*a.shape[1]+y)*a.shape[2]+z];
          if (!type) continue;
          const V world=a.lower+a.pitch*V(x,y,z);
          const double distance=solid_distance(s,inverse*(world-p.center));
          if (distance<row[type-1]) {row[type-1]=distance;piece[type-1]=k;points[type-1]=world;}
        }
    }
    if (!derivative) continue;
    for (int t=0;t<2;++t) {
      double* grad=derivative+(2*j+t)*7;
      std::fill(grad,grad+7,0.);
      if (piece[t]<0) continue;
      const auto& s=a.solids[piece[t]];
      const auto& p=poses[piece[t]];
      const V local=p.rotation.transpose()*(points[t]-p.center);
      V normal;
      for (int dim=0;dim<3;++dim) {
        const V delta=1.e-6*V::Unit(dim);
        normal[dim]=(solid_distance(s,local+delta)-solid_distance(s,local-delta))/2.e-6;
      }
      normal=p.rotation*normal;
      for (int f=s.frame;f>0;f=a.frames[f].parent) {
        const auto& link=a.frames[f];
        if (link.joint<0) continue;
        const M& z=link.skew;
        const V axis=frames[f].rotation*V(z(2,1),z(0,2),z(1,0));
        grad[link.joint]=-normal.dot(axis.cross(points[t]-frames[f].center));
      }
    }
  }
}
void* measured_arm_create(int nf,const int* parent,const int* joint,const double* origin,const double* outer,
    const double* skew,int ns,const int* link,const int* kind,const double* prim,const double* half,
    const double* lower,const int* shape,double pitch) {
  auto* arm=new Arm;
  for (int i=0;i<nf;++i) arm->frames.push_back({parent[i],joint[i],matrix_pose(origin+16*i),
      Eigen::Map<const RowM>(outer+9*i),Eigen::Map<const RowM>(skew+9*i)});
  for (int i=0;i<ns;++i) arm->solids.push_back({link[i],kind[i],matrix_pose(prim+16*i),
      Eigen::Map<const V>(half+3*i)});
  arm->lower=Eigen::Map<const V>(lower);
  arm->shape=Eigen::Map<const Eigen::Vector3i>(shape);
  arm->pitch=pitch;
  return arm;
}
void measured_arm_destroy(Arm* arm) { delete arm; }

int measured_arm_cells(const Arm* arm,const double* poses,int count,int* ids,int capacity,int* offsets) {
  const Arm& a=*arm;
  std::vector<Pose> frames(a.frames.size()),solids(a.solids.size());
  std::vector<int> marked(a.shape.prod(),-1);
  std::vector<int> cells;
  int used=0;
  for (int n=0;n<count;++n) {
    place(a,poses+7*n,frames,solids);
    cells.clear();
    offsets[n]=used;
    for (size_t i=0;i<solids.size();++i) {
      const Solid& s=a.solids[i];
      const Pose& p=solids[i];
      V bound;
      if (s.kind==0) bound=p.rotation.cwiseAbs()*s.half;
      else if (s.kind==1) {
        for (int k=0;k<3;++k)
          bound[k]=s.half[0]*std::hypot(p.rotation(k,0),p.rotation(k,1))+s.half[2]*std::abs(p.rotation(k,2));
      } else bound=s.half;
      const V relative=p.center-a.lower;
      const Eigen::Vector3i lo=((relative-bound).array()-1.0001e-8).matrix().cwiseQuotient(V::Constant(a.pitch)).array().ceil().cast<int>().matrix().cwiseMax(0);
      const Eigen::Vector3i hi=((relative+bound).array()+1.0001e-8).matrix().cwiseQuotient(V::Constant(a.pitch)).array().floor().cast<int>().matrix().cwiseMin(a.shape-Eigen::Vector3i::Ones());
      for (int x=lo[0];x<=hi[0];++x) for (int y=lo[1];y<=hi[1];++y) for (int z=lo[2];z<=hi[2];++z) {
        const int serial=(x*a.shape[1]+y)*a.shape[2]+z;
        if (marked[serial]==n) continue;
        const V local=p.rotation.transpose()*(a.lower+a.pitch*V(x,y,z)-p.center);
        if (inside(s,local)) { marked[serial]=n;cells.push_back(serial); }
      }
    }
    std::sort(cells.begin(),cells.end());
    used+=cells.size();
    if (used<=capacity) std::copy(cells.begin(),cells.end(),ids+offsets[n]);
  }
  offsets[count]=used;
  return used;
}

static void surface_gaps(const Arm* arm,const double* root,const double* qs,int n,
    const Cloud* cloud,double margin,double* output,bool exact) {
  const Arm& a=*arm;
  std::vector<Pose> frames(a.frames.size()),base(a.solids.size()),poses(a.solids.size());
  place(a,root,frames,base);
  std::vector<Pose> all(size_t(n)*a.solids.size());
  for (int j=0;j<n;++j) {
    place(a,qs+7*j,frames,poses);
    std::copy(poses.begin(),poses.end(),all.begin()+size_t(j)*poses.size());
  }
  std::fill(output,output+n,margin);
  if (cloud->nodes.empty()) return;
  std::vector<int> stack;
  stack.reserve(40);
  for (size_t cursor=0;cursor<a.solids.size();++cursor) {
    const size_t k=a.solids.size()-1-cursor;
    const auto& s=a.solids[k];
    if (s.frame==0) continue;
    std::vector<double> before(cloud->points.size(),std::numeric_limits<double>::quiet_NaN());
    for (int j=0;j<n;++j) {
      const auto& p=all[size_t(j)*poses.size()+k];
      if ((p.center-base[k].center).squaredNorm()<1.e-20 && (p.rotation-base[k].rotation).squaredNorm()<1.e-20) continue;
      const V base_bound=p.rotation.cwiseAbs()*s.half;
      const M trans=p.rotation.transpose();
      stack.assign(1,0);
      while (!stack.empty()) {
        const Node& node=cloud->nodes[stack.back()];stack.pop_back();
        const V bound=base_bound+V::Constant(std::max(0.,output[j]));
        if ((node.lo.array()>(p.center+bound).array()).any() ||
            (node.hi.array()<(p.center-bound).array()).any()) continue;
        if (node.left>=0) {
          stack.push_back(node.left);stack.push_back(node.right);continue;
        }
        for (int z=node.begin;z<node.end;++z) {
          const int id=cloud->order[z];
          const V& point=cloud->points[id];
          const V local=trans*(point-p.center);
          const double d=exact ? solid_distance(s,local) : signed_box(local,s.half);
          if (d>=output[j]) continue;
          if (std::isnan(before[id])) {
            const V initial=base[k].rotation.transpose()*(point-base[k].center);
            before[id]=(exact ? solid_distance(s,initial) : signed_box(initial,s.half))-.001;
          }
          if (d<before[id]) output[j]=d;
        }
      }
    }
  }
}
void measured_arm_surface_gaps(const Arm* arm,const double* root,const double* qs,int n,
    const Cloud* cloud,double margin,double* output) {
  surface_gaps(arm,root,qs,n,cloud,margin,output,false);
}
void measured_arm_shape_gaps(const Arm* arm,const double* root,const double* qs,int n,
    const Cloud* cloud,double margin,double* output) {
  surface_gaps(arm,root,qs,n,cloud,margin,output,true);
}
}
