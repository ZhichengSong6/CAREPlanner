#!/usr/bin/env python3
import os
import yaml
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class BodySampleMarkerPublisher:
    def __init__(self):
        self.body_samples_file = rospy.get_param('~body_samples_file')
        self.marker_topic = rospy.get_param('~marker_topic', '/care_planner/body_samples/markers')
        self.publish_rate = float(rospy.get_param('~publish_rate', 10.0))
        self.show_non_risk = bool(rospy.get_param('~show_non_risk', True))
        self.alpha = float(rospy.get_param('~alpha', 0.35))
        self.scale_multiplier = float(rospy.get_param('~scale_multiplier', 1.0))

        self.samples = self.load_samples(self.body_samples_file)
        self.pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1, latch=True)

        rospy.loginfo('[body_sample_marker_publisher] Loaded %d body samples from %s',
                      len(self.samples), self.body_samples_file)
        rospy.loginfo('[body_sample_marker_publisher] Publishing markers on %s', self.marker_topic)

    def load_samples(self, path):
        if not os.path.exists(path):
            raise RuntimeError('body_samples_file does not exist: {}'.format(path))

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        body_sampling = data.get('body_sampling', {})
        links = body_sampling.get('links', [])

        samples = []
        for link in links:
            link_name = link.get('link_name', '')
            frame = link.get('frame', link_name)
            include_for_risk = bool(link.get('include_for_risk', True))

            if (not include_for_risk) and (not self.show_non_risk):
                continue

            for i, s in enumerate(link.get('samples', [])):
                center = s.get('center', [0.0, 0.0, 0.0])
                radius = float(s.get('radius', 0.01))
                source_type = s.get('source_type', 'unknown')
                source_idx = int(s.get('source_collision_index', -1))

                samples.append({
                    'link_name': link_name,
                    'frame': frame,
                    'include_for_risk': include_for_risk,
                    'center': center,
                    'radius': radius,
                    'source_type': source_type,
                    'source_collision_index': source_idx,
                    'sample_index_in_link': i,
                })

        return samples

    def make_marker(self, sample, marker_id):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = sample['frame']
        m.ns = 'body_risk_samples'
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        c = sample['center']
        m.pose.position.x = float(c[0])
        m.pose.position.y = float(c[1])
        m.pose.position.z = float(c[2])
        m.pose.orientation.w = 1.0

        diameter = 2.0 * sample['radius'] * self.scale_multiplier
        m.scale.x = diameter
        m.scale.y = diameter
        m.scale.z = diameter

        # Color convention:
        #   cylinder-derived samples: cyan/blue
        #   box-derived samples: orange/yellow
        #   non-risk samples such as base_link: gray
        if not sample['include_for_risk']:
            m.color.r = 0.55
            m.color.g = 0.55
            m.color.b = 0.55
            m.color.a = min(self.alpha, 0.25)
        elif sample['source_type'] == 'cylinder':
            m.color.r = 0.0
            m.color.g = 0.75
            m.color.b = 1.0
            m.color.a = self.alpha
        elif sample['source_type'] == 'box':
            m.color.r = 1.0
            m.color.g = 0.55
            m.color.b = 0.0
            m.color.a = self.alpha
        else:
            m.color.r = 0.2
            m.color.g = 1.0
            m.color.b = 0.2
            m.color.a = self.alpha

        return m

    def make_label_marker(self, sample, marker_id):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = sample['frame']
        m.ns = 'body_risk_sample_labels'
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        c = sample['center']
        m.pose.position.x = float(c[0])
        m.pose.position.y = float(c[1])
        m.pose.position.z = float(c[2]) + sample['radius'] + 0.015
        m.pose.orientation.w = 1.0
        m.scale.z = 0.025

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.8
        m.text = '{}:{}'.format(sample['link_name'], sample['sample_index_in_link'])
        return m

    def publish_once(self):
        arr = MarkerArray()

        # Clear stale markers first.
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        for s in self.samples:
            arr.markers.append(self.make_marker(s, marker_id))
            marker_id += 1

        self.pub.publish(arr)

    def spin(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.publish_once()
            rate.sleep()


def main():
    rospy.init_node('body_sample_marker_publisher')
    node = BodySampleMarkerPublisher()
    node.spin()


if __name__ == '__main__':
    main()
