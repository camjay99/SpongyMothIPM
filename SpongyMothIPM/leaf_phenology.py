import torch

class LeafPhenology():
    def __init__(self,
                 config,
                 gdd_threshold=0,
                 ncd_threshold=5,
                 a=-68,
                 b=638,
                 c=-0.01):
        self.gdd_threshold = torch.tensor(gdd_threshold, 
                                          dtype=config.dtype, 
                                          requires_grad=True)
        self.ncd_threshold = torch.tensor(ncd_threshold, 
                                          dtype=config.dtype, 
                                          requires_grad=True)
        self.a = torch.tensor(a,
                              dtype=config.dtype,
                              requires_grad=True)
        self.b = torch.tensor(b,
                              dtype=config.dtype,
                              requires_grad=True)
        self.c = torch.tensor(c,
                              dtype=config.dtype,
                              requires_grad=True)

    def calc_gdd(self, temps, green_date):
        mask = (temps['yday'] >= 0) & (temps['yday'] <= green_date)
        return torch.sum(
                torch.maximum(
                    temps[mask] - self.gdd_threshold, 
                    torch.tensor(0)))

    def calc_ncd(self, temps, green_date, year):
        # Count days after Nov 1st of previous year or before green up this year
        mask = (((temps['yday'] >= 305) & (temps['year'] == year-1))
                | ((temps['yday'] <= green_date) & (temps['year'] == year)))
        return torch.count_nonzero(temps[mask] <= self.ncd_threshold)

    def calc_gcc_crit(self, ncd):
        # ncd should be a tensor
        self.gcc_crit = (self.a 
                         + (self.b
                            * torch.exp(self.c*ncd)))