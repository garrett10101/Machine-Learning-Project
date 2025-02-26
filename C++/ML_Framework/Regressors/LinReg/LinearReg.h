#ifndef LINEARREG_H
#define LINEARREG_H
#include "../MLModel_Regressor.h"
#include <vector>

class LinearReg : public MLModel_Regressor {
    public:
        LinearReg();
    
        void fit(const std::vector<double>& X, const std::vector<double>& y) override;
    
        double predict(double x) const override;
    
        void tune() override;
    
    private:
        double slope;
        double intercept;
    };
#endif // LINEARREG_H