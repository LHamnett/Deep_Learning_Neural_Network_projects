def is_necessary_restock_specific_good(current_level,
    forecasted_levels,
    minimum_level,
    desired_level,
    order_cost_fixed,
    order_cost_per_unit,
    holding_cost_per_unit):

    num_days_forecasted = len(forecasted_levels)
    for day in range(num_days_forecasted):
        future_level = forecasted_levels[day]
        if future_level < minimum_level:
            num_units_needed = minimum_level - future_level
            total_order_cost = order_cost_fixed + (num_units_needed * order_cost_per_unit)
            total_holding_cost = holding_cost_per_unit * (future_level - current_level)
            if total_order_cost < total_holding_cost:
                return True
    return False
            
            

