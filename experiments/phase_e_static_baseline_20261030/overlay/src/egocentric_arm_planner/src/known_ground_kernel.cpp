// Same rigid transforms, mesh support and joint-box radius bound as known_ground.py.
// No curve sampling, threshold or deadline decisions are made in this kernel.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
void multiply(const double* a,const double* b,double* out) {
  for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
    double value=0.;for(int k=0;k<4;++k)value+=a[4*i+k]*b[4*k+j];
    out[4*i+j]=value;
  }
}
}

static int ground_clearance_impl(
    int nf,const int32_t* parents,const int32_t* kinds,const int32_t* indices,
    const double* origins,const double* axes,const double* rotations,
    int nb,const int32_t* frames,const int32_t* starts,const double* vertices,
    const double* radii,const double* q,const double* radius,double plane,
    double* output,int32_t* output_body,const double* offsets,const double* normal=nullptr,double offset=0.) {
  if(nf<1 || nb<1 || !std::isfinite(plane))return 1;
  for(int j=0;j<7;++j)if(!std::isfinite(q[j]) || !std::isfinite(radius[j]))return 2;
  std::vector<double> transforms(static_cast<size_t>(nf)*16,0.);
  for(int j=0;j<4;++j)transforms[5*j]=1.;
  for(int i=1;i<nf;++i) {
    if(parents[i]<0 || parents[i]>=i)return 3;
    double origin_pose[16];multiply(transforms.data()+16*parents[i],origins+16*i,origin_pose);
    double move[16]={0.};for(int j=0;j<4;++j)move[5*j]=1.;
    if(kinds[i]!=0) {
      if(indices[i]<0 || indices[i]>=7)return 4;
      const double angle=q[indices[i]];
      if(kinds[i]==1) {
        const double c=std::cos(angle),s=std::sin(angle);
        for(int row=0;row<3;++row)for(int col=0;col<3;++col) {
          const int j=3*row+col;
          move[4*row+col]=c*(row==col)+(1.-c)*rotations[18*i+j]+s*rotations[18*i+9+j];
        }
      } else if(kinds[i]==2) {
        for(int j=0;j<3;++j)move[4*j+3]=axes[3*i+j]*angle;
      } else return 5;
    }
    multiply(origin_pose,move,transforms.data()+16*i);
  }
  double minimum=std::numeric_limits<double>::infinity();int32_t body=-1;
  for(int i=0;i<nb;++i) {
    if(frames[i]<0 || frames[i]>=nf || starts[i]<0 || starts[i+1]<=starts[i])return 6;
    const double* tf=transforms.data()+16*frames[i];double low=std::numeric_limits<double>::infinity();
    for(int j=starts[i];j<starts[i+1];++j) {
      const double* p=vertices+3*j;
      const double z=p[0]*tf[8]+p[1]*tf[9]+p[2]*tf[10]+tf[11];
      const double h=normal ? normal[0]*(p[0]*tf[0]+p[1]*tf[1]+p[2]*tf[2]+tf[3])+
          normal[1]*(p[0]*tf[4]+p[1]*tf[5]+p[2]*tf[6]+tf[7])+normal[2]*z+offset : z-plane;
      low=std::min(low,h);
    }
    double bound=0.;for(int j=0;j<7;++j)bound+=radii[7*i+j]*radius[j];
    low=low-bound+(offsets?offsets[i]:0.);
    if(low<minimum){minimum=low;body=i;}
  }
  if(!std::isfinite(minimum) || body<0)return 7;
  *output=minimum;*output_body=body;return 0;
}

extern "C" int known_ground_clearance(int nf,const int32_t* parents,const int32_t* kinds,const int32_t* indices,
    const double* origins,const double* axes,const double* rotations,int nb,const int32_t* frames,
    const int32_t* starts,const double* vertices,const double* radii,const double* q,const double* radius,
    double plane,double* output,int32_t* output_body) {
  return ground_clearance_impl(nf,parents,kinds,indices,origins,axes,rotations,nb,frames,starts,vertices,radii,q,radius,plane,output,output_body,nullptr);
}
extern "C" int known_ground_support_clearance(int nf,const int32_t* parents,const int32_t* kinds,const int32_t* indices,
    const double* origins,const double* axes,const double* rotations,int nb,const int32_t* frames,
    const int32_t* starts,const double* vertices,const double* radii,const double* q,const double* radius,
    double plane,double* output,int32_t* output_body,const double* offsets) {
  if(!offsets)return 8;
  for(int i=0;i<nb;++i)if(!std::isfinite(offsets[i]))return 9;
  return ground_clearance_impl(nf,parents,kinds,indices,origins,axes,rotations,nb,frames,starts,vertices,radii,q,radius,plane,output,output_body,offsets);
}

extern "C" int known_analytic_plane_clearance(int nf,const int32_t* parents,const int32_t* kinds,const int32_t* indices,
    const double* origins,const double* axes,const double* rotations,int nb,const int32_t* frames,
    const int32_t* starts,const double* vertices,const double* radii,const double* q,const double* radius,
    double plane,double* output,int32_t* output_body,const double* offsets,const double* normal,double offset) {
  if(!normal || !offsets || !std::isfinite(offset))return 8;
  double norm=0.;for(int i=0;i<3;++i){if(!std::isfinite(normal[i]))return 8;norm+=normal[i]*normal[i];}
  if(std::abs(norm-1.)>1e-10)return 8;
  for(int i=0;i<nb;++i)if(!std::isfinite(offsets[i]))return 9;
  return ground_clearance_impl(nf,parents,kinds,indices,origins,axes,rotations,nb,frames,starts,vertices,radii,q,radius,plane,output,output_body,offsets,normal,offset);
}
