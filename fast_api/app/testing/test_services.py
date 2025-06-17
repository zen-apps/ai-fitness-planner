import requests
import sys
import traceback
import app.helpers.process_users as process_users


create_user_json = {
    "username": "7kthg&",
    "metric": False,
    "height": 72,
    "weight": 176.5,
    "gender": 1,
    "birth_date_datetime": "2003-03-09T23:17:42.037Z",
    "activity_level_id": 3,
    "goal_id": 1,
    "goal_date_datetime": "2023-05-09T23:17:42.037Z",
    "bodyfat_override_bool": True,
    "bodyfat_override": 14,
    "baseline_calorie_override_bool": False,
    "baseline_calorie_override": 0,
}


def test_service(test_name, service_options, payload=None):
    service = service_options["service"]
    if test_name == "bodyfat":
        bodyfat_input_dict = service_options["payload"]
        bodyfat_input_dict["bodyfat_override_bool"] = False
        bodyfat_resp = process_users.bodyfat_perc_calc(bodyfat_input_dict)
        bodyfat_check_resp = {
            "bodyfat_perc_calc": 16,
            "bodyfat_perc": 16,
            "lbm_calc": 147,
            "lbm_rounded": 125,
        }
        bodyfat_bool = False
        if bodyfat_resp == bodyfat_check_resp:
            bodyfat_bool = True
        return {
            "test_name": test_name,
            "pass": bodyfat_bool,
            "message": None,
            "resp": bodyfat_resp,
        }
    status_code = service_options["status_code"]
    response = requests.get(service)
    stat_code = response.status_code
    try:
        assert response.status_code == status_code, "Oh no! This assertion failed!"
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        message = "Error occurred on line {} in statement {} {}".format(
            filename, line, func
        )
        # exit(1)
        return {"test_name": test_name, "pass": False, "message": message}
    else:
        return {"test_name": test_name, "pass": True, "message": None}
