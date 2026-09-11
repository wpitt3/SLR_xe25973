#!/usr/bin/env python3
import datetime as dt
import sys
from common import INSTANCE_TYPE, ec2, region_arg, tags, key_name, SG_NAME, ROLE


def get_sg(region):
    group = ec2(region).describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])["SecurityGroups"]
    if group:
        return group[0]["GroupId"]



AMIS = {
    "eu-north-1":       "ami-04483419bb184be38", #Stockholm
    "us-east-2":        "ami-05d63ba6a40bd814c", # Ohio
    "ap-northeast-2": "ami-0dade0a5aab40ac03", # Seoul
    "sa-east-1": "ami-0dfe1b5c63bdcd7e2", #Sao Paulo
    "eu-west-2": "ami-02139d0ab7c5ba481", #Landon
    "us-west-2": "ami-09b010b33f6e301e8", #Oregon
}


def get_ami(region):
    if region not in AMIS:
        sys.exit(f"No AMI recorded for {region}")
    return AMIS[region]


def max_price(c):
    history = c.describe_spot_price_history(
        InstanceTypes=[INSTANCE_TYPE], ProductDescriptions=["Linux/UNIX"],
        StartTime=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6),
    )["SpotPriceHistory"]

    return round(max(float(h["SpotPrice"]) for h in history) * 1.3, 4)


def spot_count_arg():
    if len(sys.argv) < 3:
        return 1
    return int(sys.argv[2])


def main():
    region = region_arg()
    c = ec2(region)
    spot_count = spot_count_arg()
    ami = get_ami(region)
    key = key_name(region)
    sg = get_sg(region)
    price = max_price(c)

    result = c.run_instances(
        ImageId=ami, InstanceType=INSTANCE_TYPE, KeyName=key, MinCount=spot_count, MaxCount=spot_count,
        SecurityGroupIds=[sg],
        IamInstanceProfile={"Name": ROLE},
        InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate",
            "MaxPrice": str(price)}},
        TagSpecifications=[tags("instance", "dl-spot")],
    )["Instances"]

    iids = [inst["InstanceId"] for inst in result]
    print(f"{price}/hr waiting")
    c.get_waiter("instance_running").wait(InstanceIds=iids)

    descs = c.describe_instances(InstanceIds=iids)["Reservations"]
    ips = []
    for r in descs:
        for i in r["Instances"]:
            ip = i.get("PublicIpAddress", "")
            ips.append(ip)
            print(f"Running: {i['InstanceId']}")
    for ip in ips:
        print(f"SPOT_IP={ip}")


if __name__ == "__main__":
    main()
