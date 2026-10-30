#pragma once
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace egocentric_arm_planner {
// Conservative interval propagation through ONLY the existing dynamics and
// acceleration rows. This never changes a solver bound or authorizes motion.
// A finite row upper bound below its original lower bound proves a local
// conflict, including executable rows that combine position and velocity.
inline std::vector<double> primitiveReachableUpperBounds(
    const Eigen::SparseMatrix<double>& a,const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& g,const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,const Eigen::VectorXd& xlower,
    const Eigen::VectorXd& xupper,int acceleration_rows,int first,int count) {
  const double nan=std::numeric_limits<double>::quiet_NaN();
  std::vector<double> result(std::max(0,count),nan);
  if(count==0)return result;
  if(count<0 || first<0 || first+count>g.rows() || acceleration_rows<0 ||
      acceleration_rows>g.rows() || a.cols()!=g.cols() || b.size()!=a.rows() ||
      lower.size()!=g.rows() || upper.size()!=g.rows() ||
      xlower.size()!=g.cols() || xupper.size()!=g.cols())return result;
  Eigen::VectorXd lo=xlower,hi=xupper;
  const Eigen::SparseMatrix<double,Eigen::RowMajor> ar=a,gr=g;
  auto contract=[&](const Eigen::SparseMatrix<double,Eigen::RowMajor>& m,int row,
                    double row_lo,double row_hi) {
    double sum_lo=0.,sum_hi=0.,scale=1.;
    for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(m,row);it;++it) {
      const double v=it.value();if(v==0.)continue;
      const int j=it.col();
      if(!std::isfinite(v) || !std::isfinite(lo[j]) || !std::isfinite(hi[j]) ||
          std::abs(lo[j])>=1e29 || std::abs(hi[j])>=1e29)return true;
      sum_lo+=v*(v>0.?lo[j]:hi[j]);sum_hi+=v*(v>0.?hi[j]:lo[j]);
      scale+=std::abs(v)*std::max(std::abs(lo[j]),std::abs(hi[j]));
    }
    const double pad=1e-10*scale;
    for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(m,row);it;++it) {
      const double v=it.value();if(std::abs(v)<1e-14)continue;
      const int j=it.col();
      // Outward padding prevents roundoff from manufacturing a contradictory
      // bound during repeated contraction.
      const double contribution_lo=v*(v>0.?lo[j]:hi[j]);
      const double contribution_hi=v*(v>0.?hi[j]:lo[j]);
      double nlo=lo[j],nhi=hi[j];
      if(row_lo>-1e29 && std::isfinite(row_lo)) {
        const double bound=(row_lo-(sum_hi-contribution_hi)-pad)/v;
        if(v>0.)nlo=std::max(nlo,bound);else nhi=std::min(nhi,bound);
      }
      if(row_hi<1e29 && std::isfinite(row_hi)) {
        const double bound=(row_hi-(sum_lo-contribution_lo)+pad)/v;
        if(v>0.)nhi=std::min(nhi,bound);else nlo=std::max(nlo,bound);
      }
      if(nlo>nhi)return false;
      // Update row totals so later variables still see a valid evaluation.
      sum_lo+=v*(v>0.?nlo:nhi)-contribution_lo;
      sum_hi+=v*(v>0.?nhi:nlo)-contribution_hi;
      lo[j]=nlo;hi[j]=nhi;
    }
    return true;
  };
  if((lo.array()>hi.array()).any())return result;
  for(int pass=0;pass<8;++pass) {
    for(int row=0;row<a.rows();++row)if(!contract(ar,row,b[row],b[row]))return result;
    for(int row=0;row<acceleration_rows;++row)
      if(!contract(gr,row,lower[row],upper[row]))return result;
    for(int row=a.rows()-1;row>=0;--row)if(!contract(ar,row,b[row],b[row]))return result;
  }
  for(int i=0;i<count;++i) {
    double value=0.,scale=1.;bool finite=true;
    for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(gr,first+i);it;++it) {
      if(it.value()==0.)continue;
      const double endpoint=it.value()>0.?hi[it.col()]:lo[it.col()];
      if(!std::isfinite(endpoint) || std::abs(endpoint)>=1e29) {finite=false;break;}
      value+=it.value()*endpoint;scale+=std::abs(it.value()*endpoint);
    }
    if(finite)result[i]=value+1e-10*scale;
  }
  return result;
}
}
