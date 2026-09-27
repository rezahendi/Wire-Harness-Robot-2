"""Command-line client: reset the cell (optional) and run one routing job.

    ros2 run harness_task route_harness                 # route + connector
    ros2 run harness_task route_harness --reset 7       # new layout (seed 7) first
    ros2 run harness_task route_harness --skip-connector
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from harness_interfaces.action import RouteHarness
from harness_interfaces.srv import ResetCell
from rclpy.action import ActionClient
from rclpy.node import Node


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", type=int, default=None, metavar="SEED",
                    help="rebuild the simulated cell with this seed first (-1 = random)")
    ap.add_argument("--nominal", action="store_true", help="with --reset: no randomisation")
    ap.add_argument("--skip-connector", action="store_true")
    ap.add_argument("--timeout", type=float, default=0.0)
    args, ros_args = ap.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = Node("route_harness_client")
    try:
        if args.reset is not None:
            cli = node.create_client(ResetCell, "/sim/reset")
            if not cli.wait_for_service(timeout_sec=10.0):
                node.get_logger().error("/sim/reset not available")
                return 2
            req = ResetCell.Request(seed=args.reset, randomize=not args.nominal)
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
            node.get_logger().info(fut.result().message)

        ac = ActionClient(node, RouteHarness, "route_harness")
        if not ac.wait_for_server(timeout_sec=20.0):
            node.get_logger().error("route_harness action server not available")
            return 2
        goal = RouteHarness.Goal(skip_connector=args.skip_connector, timeout=args.timeout)
        last = {"phase": None}

        def on_fb(msg):
            fb = msg.feedback
            if fb.phase != last["phase"]:
                last["phase"] = fb.phase
                where = f"fork {fb.current_fork}" if fb.current_fork >= 0 else "connector"
                node.get_logger().info(f"[{fb.elapsed:6.1f} s] {where:9s} {fb.phase:20s} "
                                       f"tension {fb.wire_tension:4.1f} N  |F| {fb.contact_force:4.1f} N")

        send = ac.send_goal_async(goal, feedback_callback=on_fb)
        rclpy.spin_until_future_complete(node, send)
        handle = send.result()
        if not handle.accepted:
            node.get_logger().error("goal rejected")
            return 1
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(node, res_fut)
        res = res_fut.result().result
        node.get_logger().info(f"success={res.success} forks={list(res.forks_routed)} "
                               f"connector_seated={res.connector_seated} time={res.duration:.1f}s: {res.message}")
        return 0 if res.success else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
