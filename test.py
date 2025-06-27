
# t = (1, (2, (3,(19,10), 4), 5), 6) 
# def max_tuple_depth(t):
#     if not isinstance(t, tuple):
#         return 0
#     if not t:
#         return 1  # tuple rỗng vẫn có độ sâu là 1
#     return 1 + max((max_tuple_depth(item) for item in t), default=0)
# mx = max_tuple_depth(t)
# print(mx)

from datetime import datetime, date, time, timedelta
import datetime as dt
date_string = "25-11-2024 14:30:00"
parse_date = datetime.strptime (date_string, "%d-%m-%Y %H:%M:%S")
print(date_string)
print(parse_date)
