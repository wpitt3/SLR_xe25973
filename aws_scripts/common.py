#!/usr/bin/env python3
import os, sys, boto3

SUFFIX = "xe25973"
SG_NAME = "ssh-anywhere"
INSTANCE_TYPE = "g5.xlarge"
TAG = {"Key": "ManagedBy", "Value": "spot-scripts"}
KEY_DIR = os.path.expanduser("/usr/local/development/SLR_xe25973/pems/")


def region_arg():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} REGION")
    return sys.argv[1]


def ec2(region):
    return boto3.client("ec2", region_name=region)


def key_name(region):
    return f"{region}_{SUFFIX}"


def key_path(region):
    return os.path.join(KEY_DIR, key_name(region) + ".pem")


def tags(resource_type, name=None):
    t = [TAG] + ([{"Key": "Name", "Value": name}] if name else [])
    return {"ResourceType": resource_type, "Tags": t}


def bucket_name(region):
    return f"mska-{region}-{SUFFIX}"


ROLE = f"spot-s3-{SUFFIX}"
BUCKET_GLOB = f"mska-*-{SUFFIX}"
