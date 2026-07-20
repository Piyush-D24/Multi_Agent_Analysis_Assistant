import sys
sys.path.insert(0, ".")

from mcp_server.tools.csv_profile_tools import mcp_profile_csv

result = mcp_profile_csv("events_sample.csv")
print(result["status"])
print(result["rows"])