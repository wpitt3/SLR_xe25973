#!/usr/bin/env python3
import boto3
from common import region_arg


def main():
    region = region_arg()
    for r in boto3.client("ec2", region_name=region).describe_instances()["Reservations"]:
        for i in r["Instances"]:
            print(region, i["InstanceId"], i["InstanceType"], i["State"]["Name"],
                  i.get("InstanceLifecycle", "on-demand"), i.get("PublicIpAddress", "-"))

if __name__ == "__main__":
    main()
