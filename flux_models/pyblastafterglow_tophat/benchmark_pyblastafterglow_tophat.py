from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowFlux



name = "tophat"
model_dir = f"./model/"
FILTERS = ["radio-3GHz", "radio-6GHz", "bessellv", "X-ray-1keV"]

lc_model = AfterglowFlux(name,
                         directory = model_dir, 
                         filters = FILTERS,
                         model_type= "MLP")
 
for metric_name in ["L2", "Linf"]:   

    
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "./model/pyblastafterglow_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()

    benchmarker.plot_lightcurves_mismatch(parameter_labels = ["$\\iota$", 
                                                              "$\log_{10}(E_0)$", 
                                                              "$\\theta_{\\mathrm{c}}$", 
                                                              "$\log_{10}(n_{\mathrm{ism}})$", 
                                                              "$p$", "$\\epsilon_E$", 
                                                              "$\\epsilon_B$",
                                                              "$\\Gamma_0$"])



