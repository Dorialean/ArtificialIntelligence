import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Построить нечёткую базу знаний для задачи определения цены видеокарты для
# компьютера ( потребность в графической обработке, электропотребление,
# разрешение экрана), проверить её на полноту и произвести нечёткий вывод для
# конкретных значений (выбрать случайным образом).

# def main():
#     matplotlib.use('TkAgg')
#
#     speed = ctrl.Antecedent(np.arange(0, 200), 'speed')
#     temperature = ctrl.Antecedent(np.arange(16, 30), 'temperature')
#     consumption = ctrl.Consequent(np.arange(5, 25), 'consumption')
#
#     consumption.automf(names=['small', 'medium', 'high'])
#
#     speed['small'] = fuzz.trapmf(speed.universe, [0, 0, 30, 60])
#     speed['medium'] = fuzz.trapmf(speed.universe, [50, 70, 120, 150])
#     speed['high'] = fuzz.trapmf(speed.universe, [110, 140, 200, 200])
#
#     temperature['low'] = fuzz.trapmf(temperature.universe, [16, 16, 20, 25])
#     temperature['high'] = fuzz.trapmf(temperature.universe, [20, 25, 30, 50])
#
#     speed.view()
#     temperature.view()
#     consumption.view()
#
#     rule1 = ctrl.Rule(speed['small'] & temperature['low'], consumption['small'])
#     rule2 = ctrl.Rule(speed['small'] & temperature['high'], consumption['small'])
#     rule3 = ctrl.Rule(speed['medium'] & temperature['low'], consumption['high'])
#     rule4 = ctrl.Rule(speed['medium'] & temperature['high'], consumption['medium'])
#     rule5 = ctrl.Rule(speed['high'] & temperature['low'], consumption['high'])
#     rule6 = ctrl.Rule(speed['high'] & temperature['high'], consumption['high'])
#
#     consumption_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
#     consumption_simulator = ctrl.ControlSystemSimulation(consumption_ctrl)
#
#     #
#     consumption_simulator.input['speed'] = 110
#     consumption_simulator.input['temperature'] = 22
#
#     #
#     consumption_simulator.compute()
#     print(consumption_simulator.output['consumption'])
#
#     speed.view(sim=consumption_simulator)
#     temperature.view(sim=consumption_simulator)
#     consumption.view(sim=consumption_simulator)
#
#     plt.show()


def main():
    graphical_need = ctrl.Antecedent(np.arange(0, 100), 'graphical_need')
    resolution = ctrl.Antecedent(np.arange(480 * 800, 4096 * 2160, 10), 'resolution')
    electricity = ctrl.Antecedent(np.arange(450, 750, 10), 'electricity')
    price = ctrl.Consequent(np.arange(500, 5750), 'price')

    price.automf(names=['low', 'medium', 'high'])

    graphical_need['low'] = fuzz.trapmf(graphical_need.universe, [0, 20, 30, 43])
    graphical_need['high'] = fuzz.trapmf(graphical_need.universe, [50, 60, 90, 100])

    resolution['low'] = fuzz.trapmf(resolution.universe, [480 * 800, 600 * 800, 580 * 900, 900 * 800])
    resolution['medium'] = fuzz.trapmf(resolution.universe, [1240 * 800, 1560 * 900, 1600 * 1000, 1920 * 1080])
    resolution['high'] = fuzz.trapmf(resolution.universe, [2000 * 2000, 2100 * 2100, 3840 * 1990, 4096 * 2160])

    electricity['low'] = fuzz.trapmf(electricity.universe, [450, 450, 500, 550])
    electricity['medium'] = fuzz.trapmf(electricity.universe, [560, 580, 590, 600])
    electricity['high'] = fuzz.trapmf(electricity.universe, [620, 650, 700, 750])

    graphical_need.view()
    resolution.view()
    electricity.view()

    rule1 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['low'], price['low'])
    rule2 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['high'], price['high'])
    rule3 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['medium'], price['low'])
    rule4 = ctrl.Rule(graphical_need['low'] & resolution['low'] & electricity['high'], price['low'])
    rule5 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['low'], price['high'])
    rule6 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['low'], price['high'])
    rule7 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['low'], price['medium'])
    rule8 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['medium'], price['medium'])
    rule9 = ctrl.Rule(graphical_need['low'] & resolution['medium'] & electricity['high'], price['medium'])
    rule10 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['low'], price['high'])
    rule11 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['medium'], price['high'])
    rule12 = ctrl.Rule(graphical_need['high'] & resolution['medium'] & electricity['high'], price['high'])
    rule13 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['medium'], price['low'])
    rule14 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['medium'], price['medium'])
    rule15 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['medium'], price['high'])
    rule16 = ctrl.Rule(graphical_need['low'] & resolution['high'] & electricity['high'], price['medium'])
    rule17 = ctrl.Rule(graphical_need['high'] & resolution['low'] & electricity['high'], price['low'])
    rule18 = ctrl.Rule(graphical_need['high'] & resolution['high'] & electricity['low'], price['high'])

    price_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15,
         rule16, rule17, rule18])
    price_simulator = ctrl.ControlSystemSimulation(price_ctrl)

    price_simulator.input['graphical_need'] = 0
    price_simulator.input['resolution'] = 1920 * 1080
    price_simulator.input['electricity'] = 600

    price_simulator.compute()
    print(price_simulator.output['price'])

    graphical_need.view(sim=price_simulator)
    resolution.view(sim=price_simulator)
    electricity.view(sim=price_simulator)

    plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
