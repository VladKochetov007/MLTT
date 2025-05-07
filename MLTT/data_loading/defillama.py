import asyncio
from defillama import DefiLlama as dfl
import pandas as pd
import requests
from json.decoder import JSONDecodeError


def _get_chains(url: str = "https://raw.githubusercontent.com/DefiLlama/DefiLlama-Adapters/refs/heads/main/projects/helper/chains.json"):
    response = requests.get(url)
    return response.json()


def _format_dfl_frame(data: list[dict], 
                      date_index: bool = True,
                      format_date: bool = True) -> pd.DataFrame:
    """
    Format the data from DefiLlama into a pandas DataFrame.


    Example input:
    [
        {'date': 1524787200, 'tvl': 404078},
        {'date': 1524873600, 'tvl': 400585},
        {'date': 1524960000, 'tvl': 429214},
        {'date': 1525046400, 'tvl': 421651},
        ...
    ]
    """
    if date_index:
        frame = pd.DataFrame(data).set_index("date")
    else:
        frame = pd.DataFrame(data)

    if format_date:
        if date_index:
            frame.index = pd.to_datetime(frame.index, unit="s")
        else:
            frame["date"] = pd.to_datetime(frame["date"], unit="s")
    return frame

def _format_slash_nan(value: str, nan_value = pd.NA) -> str:
    """
    Convert "-" to NaN.
    """
    return nan_value if value == "-" else value


class DefiLlamaAdapter:
    def __init__(self):
        self.dfl = dfl()

    def market_tvl(self) -> pd.DataFrame:
        """
        Get historical total value locked (TVL) data for a entire DeFi market from DefiLlama.

        Returns:
            - `pd.DataFrame`: DataFrame with date index and columns:
                - `tvl`: Total Value Locked
        """
        return _format_dfl_frame(
            self.dfl.get_historical_tvl()
        ).rename(columns={"tvl": "mkt_tvl"})

    def _protocols(self) -> pd.DataFrame:
        """
        Get all protocols from DefiLlama.
        """
        data = self.dfl.get_all_protocols()
        protocols_list = []
        for protocol in data:
            protocols_list.append({
                'symbol': _format_slash_nan(protocol['symbol']),
                'name': _format_slash_nan(protocol['name']),
                'slug': _format_slash_nan(protocol['slug']),
                'category': _format_slash_nan(protocol['category']),
                'chains': _format_slash_nan(protocol['chains'])
            })
        return pd.DataFrame(protocols_list)

    async def _chain_tvl(self, chain: str) -> pd.DataFrame:
        """
        Get historical total value locked (TVL) data for a given chain from DefiLlama.
        """
        try:
            response = self.dfl.get_historical_tvl_chain(chain)
            if isinstance(response, dict) and response.get("statusCode") != 200:  # Check HTTP status code first
                return pd.DataFrame()
            return _format_dfl_frame(response)
        except (JSONDecodeError, requests.exceptions.RequestException) as e:
            print(f"Error fetching data for chain {chain}: {str(e)}")
            return pd.DataFrame()

    async def chain_tvl(
            self, 
            chain: str | list[str] | None = None) -> pd.DataFrame:
        """
        Get historical total value locked (TVL) data for given chain(s).

        Args:
            - `chain` (str | list[str] | None): The chain(s) to get TVL data for. 
                Default is None, which will return all chains.
        
        Returns:
            - `pd.DataFrame`: DataFrame with date index and token 
                columns in format `<chain>.tvl`
        """
        if chain is None:
            chain = _get_chains()

        if isinstance(chain, str):
            chain = [chain]

        for i, c in enumerate(chain):
            chain[i] = c.replace(" ", "-")  # DefiLlama uses "-" for spaces in API
    
        tasks = [self._chain_tvl(c) for c in chain]
        results = await asyncio.gather(*tasks)
        result = [r.rename(columns={"tvl": f"{c}.tvl"}) for c, r in zip(chain, results)]
        return pd.concat(result, axis=1)

    def async_chain_tvl(self, chain: str | list[str] | None = None) -> pd.DataFrame:
        """
        Synchronous wrapper for `chain_tvl`
        """
        return asyncio.run(self.chain_tvl(chain))