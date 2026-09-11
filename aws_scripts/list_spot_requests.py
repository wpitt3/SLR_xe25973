#!/usr/bin/env python3
import boto3
from common import region_arg


def main():
    region = region_arg()
    for r in boto3.client("ec2", region_name=region).describe_spot_instance_requests()["SpotInstanceRequests"]:
        print(region, r["SpotInstanceRequestId"], r["State"], r["Type"],
              r.get("InstanceId", "-"), r["Status"]["Code"])


if __name__ == "__main__":
    main()
