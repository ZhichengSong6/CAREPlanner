#pragma once
#include <care_confidence_map/QueryConfidence.h>
#include <cmath>
namespace care_confidence_map {
inline bool validConfidenceResponse(const QueryConfidence& q,bool require_inside=false) {
  const auto n=q.request.points.size();
  if(!n || q.response.confidence.size()!=n || q.response.current_visibility.size()!=n || q.response.inside_map.size()!=n)return false;
  for(std::size_t i=0;i<n;++i) {
    const double c=q.response.confidence[i],v=q.response.current_visibility[i];
    if(!std::isfinite(c) || !std::isfinite(v) || c<0 || c>1 || v<0 || v>1 ||
        q.response.inside_map[i]>1 || (require_inside && !q.response.inside_map[i]))return false;
  }
  return true;
}
}  // namespace care_confidence_map
