import pandas as pd
import numpy as np

import datetime
import helpers.generic_functions as generic_functions


def baseline_calorie_calc(lbm, activity_level_id):  # katch_mccardle
    try:
        adjust_bmr_activity = 1.00
        if activity_level_id == 1:
            adjust_bmr_activity = 1.05
        elif activity_level_id == 2:
            adjust_bmr_activity = 1.15
        elif activity_level_id == 3:
            adjust_bmr_activity = 1.25
        elif activity_level_id == 4:
            adjust_bmr_activity = 1.35
        elif activity_level_id == 5:
            adjust_bmr_activity = 1.55
        baseline_calories = (370 + (9.82 * lbm)) * adjust_bmr_activity
        if baseline_calories < 800:
            return 800
        elif baseline_calories > 4000:
            return 4000
        return round(baseline_calories, 0)
    except Exception as e:
        print("baseline calc exception", e)
        return None


def lbm_calc(row):
    try:
        return (1 - row["bodyfat"]) * row["weight"]
    except Exception as e:
        print("lbm calc issue", e)
        return None


def bodyfat_perc_calc(
    row,
):  # {'metric': False, 'height': 70, 'weight': 150, 'age': 40, 'gender': 1}
    age = 40
    try:
        age = calculate_age(row["birth_date_datetime"])
        print("process_users 45", age, row["birth_date_datetime"])
    except Exception as e:
        age = calculate_age(row["birth_date_datetime"], True)
        print("process_users 45", age, row["birth_date_datetime"], e)
    print("process_users 49 final age", age, row["birth_date_datetime"])
    if row["metric"]:
        weight = row["weight"] / 0.453592  # /2.205
        height = row["height"] / 2.54  # /0.3937
    else:
        weight = row["weight"]
        height = row["height"]
    bmi = (weight / (height * height)) * 703.0

    if int(row["gender"]) == 1:
        gender = 1
    else:
        gender = 0

    bodyfat_perc_calc = ((1.29 * bmi) + (0.20 * age) - (11.40 * gender) - 8.00) / 100
    if row["bodyfat_override_bool"] == True:
        bodyfat_perc = row["bodyfat_override"] / 100
    else:  # Deurenberg formula 2 : https://globalrph.com/medcalcs/estimation-of-total-body-fat/
        bodyfat_perc = bodyfat_perc_calc
    lbm_estimate = lbm_calc({"bodyfat": bodyfat_perc, "weight": weight})

    lbm_ranges = [-np.inf, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    for l in lbm_ranges:
        if lbm_estimate > l:
            # lbm_rounded = round(lbm_estimate / 25, 0) * 25
            lbm_rounded = l
    if lbm_rounded == -np.inf:
        lbm_rounded = lbm_ranges[1]
    if bodyfat_perc < 0:
        bodyfat_perc = 0.0
    elif bodyfat_perc > 0.75:
        bodyfat_perc = 0.75
    if lbm_estimate < 50:
        lbm_estimate = 50.0
    elif lbm_estimate > 250.0:
        lbm_estimate = 250.0
    return {
        "bodyfat_perc_calc": round((bodyfat_perc_calc * 100), 0),
        "bodyfat_perc": round((bodyfat_perc * 100), 0),
        "lbm_calc": round(lbm_estimate, 0),
        "lbm_rounded": round(lbm_rounded, 0),
    }


def baseline_calorie_override_check(row):
    try:
        if row["baseline_calorie_override_bool"]:
            return row["baseline_calorie_override"]
        else:
            return row["baseline_calorie_calc"]
    except Exception as e:
        print("baseline calorie calc issue", e)
        return None


def bodyfat_perc_override_check(row):
    try:
        if row["bodyfat_override_bool"]:
            return row["bodyfat_override"]
        else:
            return row["bodyfat_calc"]
    except Exception as e:
        print("bodyfat calc issue", e)
        return None


def lbm_calc(row):
    try:
        return (1 - row["bodyfat"]) * row["weight"]
    except Exception as e:
        print("lbm calc issue", e)
        return None


def calculate_age(born, convert_from_ISO=False):
    today = datetime.date.today()
    if convert_from_ISO:
        born = datetime.datetime.fromisoformat(born).date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def katch_mccardle(lbm, activity_level_id):
    adjust_bmr_activity_range_list = [1.00, 1.00]
    if activity_level_id == 1:
        adjust_bmr_activity_range_list = [1.05, 1.20]
    elif activity_level_id == 2:
        adjust_bmr_activity_range_list = [1.10, 1.375]
    elif activity_level_id == 3:
        adjust_bmr_activity_range_list = [1.20, 1.55]
    elif activity_level_id == 4:
        adjust_bmr_activity_range_list = [1.35, 1.725]
    elif activity_level_id == 5:
        adjust_bmr_activity_range_list = [1.50, 1.90]
    round_total_calories = 50
    baseline_calories_bbdt = (
        round(
            (
                ((370 + (9.82 * lbm)) * adjust_bmr_activity_range_list[0])
                / round_total_calories
            ),
            0,
        )
        * round_total_calories
    )
    baseline_calories_katch = (
        round(
            (
                ((370 + (9.82 * lbm)) * adjust_bmr_activity_range_list[1])
                / round_total_calories
            ),
            0,
        )
        * round_total_calories
    )
    output_dict = {
        "lbm": lbm,
        "activity_level_id": activity_level_id,
        "baseline_bbdt": baseline_calories_bbdt,
        "baseline_katch": baseline_calories_katch,
    }
    return pd.DataFrame(output_dict, index=[0])


def macro_detail(
    lbm,  # yes to get baseline katch
    baseline_bbdt,  # yes
    baseline_katch,  # fyi weight loss
    activity_level_id,  # yes to get baseline katch
    nutrient_mix,  # yes
    weeks_out=None,  # yes
    goal_id=3,  # yes
):
    if weeks_out is not None and weeks_out <= 3 and goal_id != 3:
        baseline_bbdt = baseline_bbdt * 0.90
    elif weeks_out is not None and weeks_out <= 6 and goal_id != 3:
        baseline_bbdt = baseline_bbdt * 0.95
    cal_rounded_factor = 50
    baseline_bbdt = round(baseline_bbdt / cal_rounded_factor, 0) * cal_rounded_factor
    carbs = round(((baseline_bbdt * nutrient_mix["carb_perc"]) / 4) / 25, 0) * 25
    protein = round(((baseline_bbdt * nutrient_mix["protein_perc"]) / 4) / 25, 0) * 25
    fat = round(((baseline_bbdt * nutrient_mix["fat_perc"]) / 9) / 10, 0) * 10

    total_rounded_calories = (
        round(((carbs * 4 + protein * 4 + fat * 9) / cal_rounded_factor), 0)
        * cal_rounded_factor
    )
    estimated_daily_deficit = total_rounded_calories - baseline_katch
    daily_category = nutrient_mix["category"]
    output_dict = {
        # "lbm": lbm,
        # "activity_level_id": activity_level_id,
        "weeks_out": weeks_out,
        # "goal_id": goal_id,
        "baseline_bbdt": baseline_bbdt,
        "baseline_katch": baseline_katch,
        "daily_category": daily_category,
        "carbs": carbs,
        "protein": protein,
        "fat": fat,
        "rounded_calories": total_rounded_calories,
        "estimated_daily_deficit": estimated_daily_deficit,
    }
    return pd.DataFrame(output_dict, index=[0])


def get_marcos_by_day(data_dict, tz):
    # data_dict = data_from_api.dict()
    df = pd.DataFrame(data_dict, index=[0])

    # for col in df.columns:
    #    if df[col].dtype == "datetime64[ns]":
    #        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    lbm = df["lbm"][0]
    activity_level_id = df["activity_level_id"][0]
    baseline_calorie = df["baseline_calorie"][0]
    goal_date_datetime = df["goal_date_datetime"][0]
    if goal_date_datetime is not None:
        goal_date_datetime = datetime.datetime.strptime(
            goal_date_datetime, "%Y-%m-%dT%H:%M:%S.%f"
        )
        goal_date_datetime_adj_tz = generic_functions.apply_tz(goal_date_datetime, tz)[
            "dt"
        ]
    goal_id = df["goal_id"][0]
    datetime_diet_day = df["datetime_diet_day"][0]
    day_week = datetime_diet_day.weekday()
    macro_date = df["datetime_diet_day"][0].date()
    try:
        weeks_out = int(((goal_date_datetime_adj_tz - datetime_diet_day).days) / 7)
    except Exception as e:
        print("233 weeks out exception", e)
        print(goal_date_datetime, datetime_diet_day)
        weeks_out = -1
    if weeks_out < 0:
        weeks_out = None

    nutrient_mix_list = [
        {"category": "low", "carb_perc": 0.10, "protein_perc": 0.40, "fat_perc": 0.50},
        {
            "category": "medium",
            "carb_perc": 0.35,
            "protein_perc": 0.30,
            "fat_perc": 0.35,
        },
        {"category": "high", "carb_perc": 0.50, "protein_perc": 0.25, "fat_perc": 0.15},
    ]
    nutrient_dict = {}
    if day_week in [0, 1]:
        nutrient_dict = nutrient_mix_list[0]
    elif day_week in [2, 3, 4]:
        nutrient_dict = nutrient_mix_list[1]
    else:
        nutrient_dict = nutrient_mix_list[2]

    baseline_katch = katch_mccardle(lbm, activity_level_id)["baseline_katch"]
    output_df = macro_detail(
        lbm,
        baseline_calorie,
        baseline_katch,
        activity_level_id,
        nutrient_dict,
        weeks_out,
        goal_id,
    )
    return output_df


def summarize_two_weeks(payload, tz, days_begin=-6, days_end=7):
    summary_df = pd.DataFrame()
    today_df = pd.DataFrame()
    datetime_diet_day = payload["datetime_diet_day"]

    for d in range(days_begin, days_end): # -6, 7 for this week (14 days ttl).  28 for next 4 wks
        adj_datetime_diet_day = datetime_diet_day + datetime.timedelta(days=d)
        lbm = payload["lbm"]
        activity_level_id = payload["activity_level_id"]
        baseline_calorie = payload["baseline_calorie"]
        goal_id = payload["goal_id"]
        goal_date_datetime = payload["goal_date_datetime"]
        payload_auto = {
            "lbm": lbm,
            "activity_level_id": activity_level_id,
            "baseline_calorie": baseline_calorie,
            "goal_id": goal_id,
            "goal_date_datetime": goal_date_datetime,
            "datetime_diet_day": adj_datetime_diet_day,
        }
        df = get_marcos_by_day(payload_auto, tz)
        df["day_position"] = d
        df["diet_date"] = str(adj_datetime_diet_day.date())
        if d == 0:
            today_df = df.copy()
        summary_df = pd.concat([summary_df, df])
        settings_dict = {
            "min_carbs": summary_df["carbs"].min(),
            "max_carbs": summary_df["carbs"].max(),
            "min_protein": summary_df["protein"].min(),
            "max_protein": summary_df["protein"].max(),
            "min_fat": summary_df["fat"].min(),
            "max_fat": summary_df["fat"].max(),
        }
        settings_df = pd.DataFrame(settings_dict, index=[0])
    return {
        "summary_df": summary_df,
        "settings_df": settings_df,
        "today_goal_df": today_df,
    }


def summarize_two_weeks_tracked(all_tracked_df, datetime_diet_day):
    summary_df = pd.DataFrame()
    today_df = pd.DataFrame()

    for d in range(-6, 1):
        adj_datetime_diet_day = datetime_diet_day + datetime.timedelta(days=d)
        tmw = d + 1
        next_day = datetime_diet_day + datetime.timedelta(days=tmw)
        all_tracked_df["datetime_diet_day"] = pd.to_datetime(
            all_tracked_df["datetime_diet_day"]
        )
        df = all_tracked_df[
            (
                all_tracked_df["datetime_diet_day"].dt.date
                >= adj_datetime_diet_day.date()
            )
            & (all_tracked_df["datetime_diet_day"].dt.date < next_day.date())
        ].copy()
        df["total_cals"] = df["calories"] * df["servings"]
        df["total_carbs"] = df["carbs"] * df["servings"]
        df["total_protein"] = df["protein"] * df["servings"]
        df["total_fat"] = df["fat"] * df["servings"]
        total_cals = df["total_cals"].sum()
        total_carbs = df["total_carbs"].sum()
        total_protein = df["total_protein"].sum()
        total_fat = df["total_fat"].sum()
        output_dict = {
            "diet_date": str(adj_datetime_diet_day.date()),
            "calories": total_cals,
            "carbs": total_carbs,
            "protein": total_protein,
            "fat": total_fat,
            "day_position": d,
        }
        output_df = pd.DataFrame(output_dict, index=[0])
        if d == 0:
            today_df = pd.concat([today_df, output_df])
        summary_df = pd.concat([summary_df, output_df])
        settings_dict = {
            "min_carbs": summary_df["carbs"].min(),
            "max_carbs": summary_df["carbs"].max(),
            "min_protein": summary_df["protein"].min(),
            "max_protein": summary_df["protein"].max(),
            "min_fat": summary_df["fat"].min(),
            "max_fat": summary_df["fat"].max(),
        }
        settings_df = pd.DataFrame(settings_dict, index=[0])
    return {
        "summary_df": summary_df,
        "settings_df": settings_df,
        "today_tracked_df": today_df,
    }


def get_graph_settings(plan_settings_df, tracked_settings_df):
    all_settings_df = pd.concat([plan_settings_df, tracked_settings_df])
    settings_dict = {
        "min_carbs": all_settings_df["min_carbs"].min(),
        "max_carbs": all_settings_df["max_carbs"].max(),
        "min_protein": all_settings_df["min_protein"].min(),
        "max_protein": all_settings_df["max_protein"].max(),
        "min_fat": all_settings_df["min_fat"].min(),
        "max_fat": all_settings_df["max_fat"].max(),
    }
    settings_df = pd.DataFrame(settings_dict, index=[0])
    return settings_df


def summarize_day(goal_df, tracked_df, user_category_id):
    carbs_actual = tracked_df["carbs"].iloc[0]
    protein_actual = tracked_df["protein"].iloc[0]
    fat_actual = tracked_df["fat"].iloc[0]
    calories_actual = tracked_df["calories"].iloc[0]

    carbs_goal = goal_df["carbs"].iloc[0]
    protein_goal = goal_df["protein"].iloc[0]
    fat_goal = goal_df["fat"].iloc[0]
    calories_goal = (carbs_goal + protein_goal) * 4 + fat_goal * 9

    calorie_deficit_low = calories_goal - goal_df["baseline_katch"].iloc[0]
    calorie_deficit_high = calories_goal - goal_df["baseline_bbdt"].iloc[0]
    estimated_calorie_deficit = (calorie_deficit_low + calorie_deficit_high) / 2

    weeks_out = goal_df["weeks_out"].iloc[0]
    if weeks_out is None:
        weeks_out_str = "Please add Goal Date in Settings"
    elif weeks_out >= 0:
        weeks_out_str = str(weeks_out) + " weeks until Goal Date"
    else:
        weeks_out_str = "Please add a new Goal Date in Settings"
    if user_category_id is None:
        user_category_id = 1
    if user_category_id > 1:
        output_dict = {
            "weeks_out": weeks_out_str,
            "carbs_actual": carbs_actual,
            "carbs_goal": carbs_goal,
            "protein_actual": protein_actual,
            "protein_goal": protein_goal,
            "fat_actual": fat_actual,
            "fat_goal": fat_goal,
            "calories_actual": calories_actual,
            "calories_goal": calories_goal,
            "goal_estimated_calorie_deficit": estimated_calorie_deficit,
        }
    else:
        output_dict = {
            "weeks_out": weeks_out_str,
            "carbs_actual": carbs_actual,
            "carbs_goal": carbs_goal,
            "protein_actual": protein_actual,
            "protein_goal": protein_goal,
            "fat_actual": fat_actual,
            "fat_goal": fat_goal,
            "calories_actual": calories_actual,
            "calories_goal": calories_goal,
        }
    today_df = pd.DataFrame(output_dict, index=[0])
    return today_df


# not used yet, for reference.  category 1 (guest), 2 login, 3 skus
def get_summary_feature_config():
    settings_dict = {
        "feature": "goal_estimated_calorie_deficit",
        "user_category_id_min": 2,
        "user_category_id_max": 99,
    }
    settings_df = pd.DataFrame(settings_dict, index=[0])
    return settings_df


def summarize_weight_tracked(user_history_df, datetime_diet_day):
    summary_df = pd.DataFrame()
    starting_weight = user_history_df["weight"].iloc[0]
    previous_tracked_weight = 0
    for d in range(-6, 1):
        adj_datetime_diet_day = datetime_diet_day + datetime.timedelta(days=d)
        tmw = d + 1
        next_day = datetime_diet_day + datetime.timedelta(days=tmw)
        user_history_df["update_datetime"] = pd.to_datetime(
            user_history_df["update_datetime"]
        )
        if d == -6:
            df = user_history_df[
                user_history_df["update_datetime"].dt.date
                < adj_datetime_diet_day.date()
            ].copy()
            df = df.sort_values("update_datetime", ascending=False)
            try:
                previous_tracked_weight = df["weight"].iloc[0]
            except:
                previous_tracked_weight = starting_weight
        df = user_history_df[
            (user_history_df["update_datetime"].dt.date >= adj_datetime_diet_day.date())
            & (user_history_df["update_datetime"].dt.date < next_day.date())
        ].copy()
        df = df.sort_values("update_datetime", ascending=False)
        try:
            updated_weight = df["weight"].iloc[0]
        except:
            updated_weight = previous_tracked_weight
        output_dict = {
            "diet_date": str(adj_datetime_diet_day.date()),
            "weight": updated_weight,
            "day_position": d,
        }
        output_df = pd.DataFrame(output_dict, index=[0])
        summary_df = pd.concat([summary_df, output_df])

        tracked_weight_list = [starting_weight] + summary_df["weight"].to_list()
        min_weight_graph_adj = min(tracked_weight_list) - 10
        max_weight_graph_adj = max(tracked_weight_list) + 10

    return {
        "starting_weight": starting_weight,
        "weight_tracking_df": summary_df,
        "weight_min_graph": min_weight_graph_adj,
        "weight_max_graph": max_weight_graph_adj,
    }


def macro_plan_by_meal(df):
    df["day_name"] = pd.to_datetime(df["diet_date"]).dt.day_name()
    df["day_number"] = pd.to_datetime(df["diet_date"]).dt.dayofweek
    df["week_number"] = pd.to_datetime(df["diet_date"]).dt.isocalendar().week
    current_week = df[df["day_position"] == 0]["week_number"].iloc[0]
    df = df[df["week_number"] == current_week]

    protein_round_factor = 5

    meal_macros_list = []
    for index, row in df.iterrows():
        day_num = row["day_number"]

        if row["baseline_bbdt"] > 3000:
            carb_meal_increment = 75
        elif row["baseline_bbdt"] > 1350:
            carb_meal_increment = 50
        else:
            carb_meal_increment = 25

        total_carbs = row["carbs"]
        carbs_day_list = []
        running_carbs_total = 0
        for meal_num in range(1, 7):
            if running_carbs_total < total_carbs:
                remaining_carbs = total_carbs - running_carbs_total
                if remaining_carbs < carb_meal_increment:
                    carbs_day_list.append(remaining_carbs)
                else:
                    carbs_day_list.append(carb_meal_increment)
                running_carbs_total += carb_meal_increment
            else:
                carbs_day_list.append(0)

        avg_protein = row["protein"] / 6 + protein_round_factor
        protein_rounded = (
            round((avg_protein / protein_round_factor), 0) * protein_round_factor
        )
        protein_day_list = [protein_rounded] * 6

        avg_fat = row["fat"] / 6
        fat_rounded = round(avg_fat)
        fat_day_list = [fat_rounded] * 6
        day_macro_dict = {
            "day_index": day_num,
            "carbs": carbs_day_list,
            "protein": protein_day_list,
            "fat": fat_day_list,
        }
        meal_macros_list.append(day_macro_dict)

    return meal_macros_list
