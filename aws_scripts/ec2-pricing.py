#!/usr/bin/env python3

from time import sleep
import boto3
from datetime import datetime, timedelta
from common import INSTANCE_TYPE

REGIONS = [
    "us-east-1", "us-east-2", "us-west-2",
    "ca-central-1",
    "eu-west-1", "eu-west-2",
    "eu-central-1", "eu-north-1",
    "ap-northeast-1", "ap-northeast-2",
    "ap-southeast-2",
    "ap-south-1",
    "sa-east-1"
]

region_to_prices = {}

for region in REGIONS:
    ec2 = boto3.client("ec2", region_name=region)

    offerings = ec2.describe_instance_type_offerings(
        LocationType="region",
        Filters=[{"Name": "instance-type", "Values": [INSTANCE_TYPE]}],
    )
    available = len(offerings["InstanceTypeOfferings"]) > 0

    if available:
        history = ec2.describe_spot_price_history(
            InstanceTypes=[INSTANCE_TYPE],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.utcnow() - timedelta(hours=1),
        )

        sph = history["SpotPriceHistory"]
        if sph:
            maxPrice = 0.0
            az_prices = {}
            for i in range(0, len(sph)):
                price = float(sph[i]["SpotPrice"])
                az = sph[i]["AvailabilityZone"]
                az_prices[az] = price
                if maxPrice < price:
                    maxPrice = price

            region_to_prices[region] = {"maxPrice": maxPrice, "az_prices": az_prices}
    sleep(0.1)

sorted_regions = dict(sorted(region_to_prices.items(), key=lambda item: item[1]["maxPrice"]))

for (region, prices) in sorted_regions.items():
    print(f"\n{region} {prices['maxPrice']}")
    for (az, price) in prices['az_prices'].items():
        print(f"{region:<18}${float(price):.4f}/hr  ({az})")
