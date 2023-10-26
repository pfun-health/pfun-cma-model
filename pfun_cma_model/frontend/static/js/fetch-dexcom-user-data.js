async function fetchDexcomData(accessToken, ctx, cb) {
  const dateFormat = (date) => {
    return `${date.getFullYear()}-${(date.getMonth() + 1)}-${(date.getDate())}T${(date.getHours())}:${(date.getMinutes())}:${(date.getSeconds())}`;
  };

  const baseURL = "https://api.dexcom.com";
  let profile = {};
  let today = null;

  try {
    const resp = await fetch(`${baseURL}/v3/users/self/dataRange`, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    });
    const dataRange = JSON.parse(await resp.text());
    today = new Date(dataRange.egvs.end.systemTime);
  } catch (error) {
    console.error(error);
    profile.errors = [`Error fetching data range: ${error}`];
    today = new Date();
  }

  const tenDaysAgo = new Date(today - 10 * 24 * 60 * 60 * 1000);
  const startDate = dateFormat(tenDaysAgo);
  const endDate = dateFormat(today);
  const dateParams = new URLSearchParams({
    startDate: startDate,
    endDate: endDate
  }).toString();

  try {
    const responses = await Promise.all([
      fetch(`${baseURL}/v3/users/self/devices`, {
        headers: {
          Authorization: `Bearer ${accessToken}`
        }
      }),
      fetch(`${baseURL}/v3/users/self/alerts?`, {
        headers: {
          Authorization: `Bearer ${accessToken}`
        }
      }),
      fetch(`${baseURL}/v3/users/self/egvs?startDate=${startDate}&endDate=${endDate}`, {
        headers: {
          Authorization: `Bearer ${accessToken}`
        }
      })
    ]);

    const data = await Promise.all(responses.map(response => response.json()));

    profile.data = {
      devices: data[0],
      alerts: data[1],
      egvs: data[2]
    };
    profile.provider = 'Dexcom';
    profile.id = data[0].userId;
    profile.displayDevice = data[0].records[0].displayDevice;
    cb(null, profile);
  } catch (error) {
    cb(error);
  }
}
