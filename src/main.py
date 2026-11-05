
#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?                 ______  ____  _______  _____
#?                / __ \ \/ /  |/  / __ \/ ___/
#?               / /_/ /\  / /|_/ / / / /\__ \
#?              / ____/ / / /  / / /_/ /___/ /
#?             /_/     /_/_/  /_/\____//____/
#?
#? Name:        main.py
#? Purpose:     Main entry point for running simulations using the pymos models
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
# import LV_13_BSIM
import LV_1_Shichman_Hodges
# import LV_28_BSIM_derivative
# import LV_39_BSIM2
# import LV_47_BSIM3v2
import LV_49_BSIM3v3
import BSIM3v3_2
# import simple_model
from Plot import MOSFETModelComparer
#? -------------------------------------------------------------------------------
points = 10     
Vgs_values = np.linspace(0, 20, points)
Vds_values = np.linspace(0, 10, points)
T_values   = np.linspace(300, 400.0, points)

# LV_1_Shichman_Hodges_PATH   = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_1_Shichman_Hodges.csv"
# LV_13_BSIM_PATH             = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_13_BSIM.csv"
# LV_28_BSIM2_mod_PATH        = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_28_BSIM2_mod.csv"
# LV_39_BSIM2_PATH            = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_39_BSIM2.csv"
# LV_47_BSIM3v2_PATH          = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_47_BSIM3v2.csv"
# LV_49_BSIM3v3_PATH          = r"D:\WORKSPACE\BSIM-Python-Implementation\data\LV_49_BSIM3v3.csv"
# simple_model_PATH           = r"D:\WORKSPACE\BSIM-Python-Implementation\data\simple_model.csv"
BSIM3v3_2_PATH                   = r"D:\WORKSPACE\0017_bsim\data\BSIM3v3_2.csv"
PLOT                        = True
# LV_1_Shichman_Hodges        = LV_1_Shichman_Hodges.ShichmanHodgesModel()
# LV_13_BSIM                  = LV_13_BSIM.BSIMLevel13Model()
# LV_28_BSIM2_mod             = LV_28_BSIM_derivative.BSIM_Model()
# LV_39_BSIM2                 = LV_39_BSIM2.BSIM2Model()
# LV_47_BSIM3v2               = LV_47_BSIM3v2.BSIM3v2_Model()
# LV_49_BSIM3v3               = LV_49_BSIM3v3.BSIM3v3_Model()
# simple_model                = simple_model.simpleModel()
BSIM3v3_2_model            = BSIM3v3_2.BSIM3v3_Model()
#? -------------------------------------------------------------------------------
def simulate_model(model, T_values, Vgs_values, Vds_values, path):
    records         = []
    combinations    = [(T, Vgs, Vds) for T in T_values for Vgs in Vgs_values for Vds in Vds_values]
    total_points    = len(combinations)

    for i, (T, Vgs, Vds) in enumerate(combinations):
        Id  = model.compute(Vgs, Vds,0.0,T)
        records.append({
            'time'  : i // total_points ,
            'T'     : T                 ,
            'VGS'   : Vgs               ,
            'VDS'   : Vds               ,
            'ID'    : Id                
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

def main():

    # simulate_model(LV_1_Shichman_Hodges     , T_values, Vgs_values, Vds_values, LV_1_Shichman_Hodges_PATH)
    # simulate_model(LV_13_BSIM               , T_values, Vgs_values, Vds_values, LV_13_BSIM_PATH          )
    # simulate_model(LV_28_BSIM2_mod          , T_values, Vgs_values, Vds_values, LV_28_BSIM2_mod_PATH     )
    # simulate_model(LV_39_BSIM2              , T_values, Vgs_values, Vds_values, LV_39_BSIM2_PATH         )
    # simulate_model(LV_47_BSIM3v2            , T_values, Vgs_values, Vds_values, LV_47_BSIM3v2_PATH       )
    # simulate_model(LV_49_BSIM3v3            , T_values, Vgs_values, Vds_values, LV_49_BSIM3v3_PATH       )
    # simulate_model(simple_model             , T_values, Vgs_values, Vds_values, simple_model_PATH        )
    simulate_model(BSIM3v3_2_model            , T_values, Vgs_values, Vds_values, BSIM3v3_2_PATH       )


    if PLOT:
        plotter = MOSFETModelComparer([
                                    #    LV_1_Shichman_Hodges_PATH, 
                                    #    LV_13_BSIM_PATH          , 
                                    #    LV_28_BSIM2_mod_PATH     ,
                                    #    LV_39_BSIM2_PATH         ,
                                    #    LV_47_BSIM3v2_PATH       ,
                                    #    LV_49_BSIM3v3_PATH       ,
                                       BSIM3v3_2_PATH
                                    #    simple_model_PATH
                                       ])
        plotter.plot()
#? -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#? -------------------------------------------------------------------------------