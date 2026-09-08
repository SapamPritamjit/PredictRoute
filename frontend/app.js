/* PredictRoute frontend — two-view routing UX */

(function () {
  "use strict";

  // ── Leaflet marker icon helper ─────────────────────────────
  function makeIcon(color) {
    var size = 28;
    var html = '<div style="width:' + size + 'px;height:' + size + 'px;">' +
      '<svg viewBox="0 0 24 24" width="' + size + '" height="' + size + '" xmlns="http://www.w3.org/2000/svg">' +
      '<path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="' + color + '"/>' +
      '<circle cx="12" cy="9" r="2.5" fill="white"/></svg></div>';
    return L.divIcon({
      html: html,
      className: 'route-marker',
      iconSize: [size, size],
      iconAnchor: [size / 2, size],
      popupAnchor: [0, -size + 2]
    });
  }

  var ORIGIN_ICON = makeIcon('#16a34a');
  var DEST_ICON = makeIcon('#dc2626');

  // ── DOM refs ─────────────────────────────────────────────────
  var $ = function (sel) { return document.querySelector(sel); };

  // Planning view
  var planningView = $("#planning-view");
  var btnLocation = $("#btn-location");
  var btnDemo = $("#btn-demo");
  var originStatus = $("#origin-status");
  var originHint = $("#origin-hint");
  var searchInput = $("#search-input");
  var searchSpinner = $("#search-spinner");
  var resultsList = $("#results-list");
  var destStatus = $("#dest-status");
  var summaryOrigin = $("#summary-origin");
  var summaryDest = $("#summary-dest");
  var datetimeInput = $("#datetime-input");
  var btnRoute = $("#btn-route");
  var routeStatus = $("#route-status");

  // Navigation view
  var navigationView = $("#navigation-view");
  var btnBack = $("#btn-back");
  var navBottomCard = $("#nav-bottom-card");
  var gpsStatusEl = $("#gps-status");

  // ── Leaflet nav map (deferred init) ──────────────────────────
  var navMap = null;
  var navRouteLayer = null;
  var navMapInitialized = false;

  function initNavMap() {
    if (navMapInitialized) {
      navMap.invalidateSize();
      return;
    }
    navMap = L.map("nav-map", {
      zoomControl: true,
      attributionControl: true
    }).setView([28.6139, 77.2090], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19
    }).addTo(navMap);
    navRouteLayer = L.layerGroup().addTo(navMap);
    navMap.on("dragstart", function () { gpsFollowing = false; });
    navMapInitialized = true;
  }

  // ── State ────────────────────────────────────────────────────
  var origin = null;
  var originType = null;
  var destination = null;
  var searchTimeout = null;
  var lastRouteData = null;

  var REPLAY_START = "2024-08-11";

  // ── GPS live tracking state ──────────────────────────────────
  var gpsWatchId = null;
  var userPositionMarker = null;
  var accuracyCircle = null;
  var gpsFollowing = true;
  var lastGpsPosition = null;

  // ── Blue user-position marker + accuracy circle ──────────────
  function createOrUpdateUserMarker(lat, lng, accuracy) {
    if (!navMap) return;

    if (!userPositionMarker) {
      userPositionMarker = L.circleMarker([lat, lng], {
        radius: 8,
        color: '#ffffff',
        weight: 3,
        fillColor: '#2563eb',
        fillOpacity: 1
      }).addTo(navMap);
      userPositionMarker.bindPopup('<div class="popup-label">Your location</div>');
    } else {
      userPositionMarker.setLatLng([lat, lng]);
    }

    if (typeof accuracy === "number" && accuracy > 0 && accuracy < 1000) {
      if (!accuracyCircle) {
        accuracyCircle = L.circle([lat, lng], {
          radius: accuracy,
          color: '#2563eb',
          weight: 1,
          fillColor: '#2563eb',
          fillOpacity: 0.08,
          opacity: 0.3
        }).addTo(navMap);
      } else {
        accuracyCircle.setLatLng([lat, lng]);
        accuracyCircle.setRadius(accuracy);
      }
    }
  }

  function removeUserMarker() {
    if (userPositionMarker && navMap) {
      navMap.removeLayer(userPositionMarker);
      userPositionMarker = null;
    }
    if (accuracyCircle && navMap) {
      navMap.removeLayer(accuracyCircle);
      accuracyCircle = null;
    }
  }

  // ── GPS live tracking start / stop ───────────────────────────
  function startGpsTracking() {
    if (!navigator.geolocation) return;
    stopGpsTracking();
    gpsFollowing = true;

    if (gpsStatusEl) {
      gpsStatusEl.textContent = "Updating your location\u2026";
      gpsStatusEl.className = "gps-status gps-active";
    }

    gpsWatchId = navigator.geolocation.watchPosition(
      function (pos) {
        var lat = pos.coords.latitude;
        var lng = pos.coords.longitude;
        var accuracy = pos.coords.accuracy;

        lastGpsPosition = { lat: lat, lng: lng, accuracy: accuracy };

        if (navMap) {
          createOrUpdateUserMarker(lat, lng, accuracy);

          if (gpsFollowing) {
            navMap.panTo([lat, lng], { animate: true, duration: 0.5 });
          }
        }

        if (gpsStatusEl) {
          gpsStatusEl.textContent = "Live location active";
          gpsStatusEl.className = "gps-status gps-active";
        }
      },
      function (err) {
        if (gpsStatusEl) {
          gpsStatusEl.textContent = "Live location unavailable";
          gpsStatusEl.className = "gps-status gps-error";
        }
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 }
    );
  }

  function stopGpsTracking() {
    if (gpsWatchId !== null) {
      navigator.geolocation.clearWatch(gpsWatchId);
      gpsWatchId = null;
    }
    removeUserMarker();
    gpsFollowing = true;
    lastGpsPosition = null;
    if (gpsStatusEl) {
      gpsStatusEl.className = "gps-status hidden";
    }
  }

  // ── View switching ───────────────────────────────────────────
  function showNavigation() {
    planningView.classList.add("hidden");
    navigationView.classList.remove("hidden");
    initNavMap();
    setTimeout(function () { if (navMap) navMap.invalidateSize(); }, 100);
  }

  function showPlanning() {
    navigationView.classList.add("hidden");
    planningView.classList.remove("hidden");
  }

  // ── Init departure to current local hour (rounded down) ──────
  function initDatetime() {
    var now = new Date();
    var y = now.getFullYear();
    var m = now.getMonth() + 1;
    var d = now.getDate();
    var h = now.getHours();
    var todayStr = y + "-" + String(m).padStart(2, "0") + "-" + String(d).padStart(2, "0");
    if (todayStr < REPLAY_START) { y = 2024; m = 8; d = 11; h = 8; }
    var pad = function (n) { return String(n).padStart(2, "0"); };
    datetimeInput.value = y + "-" + pad(m) + "-" + pad(d) + "T" + pad(h) + ":00";
  }
  initDatetime();

  // ── Helpers ──────────────────────────────────────────────────
  function showStatus(el, msg, type) {
    el.textContent = msg;
    el.className = "status " + type;
  }
  function hideStatus(el) { el.className = "status hidden"; }
  function escHtml(s) {
    var d = document.createElement("div");
    d.appendChild(document.createTextNode(s));
    return d.innerHTML;
  }

  function setOrigin(lat, lon, label, type) {
    origin = { lat: lat, lon: lon, label: label };
    originType = type || "REAL";
    var display = label;
    if (type === "DEMO") {
      display += ' <span class="demo-badge">DEMO</span>';
    }
    summaryOrigin.innerHTML = display;
    btnRoute.disabled = !(origin && destination);
  }

  function setDestination(lat, lon, label, displayName) {
    destination = { lat: lat, lon: lon, label: label, displayName: displayName };
    summaryDest.textContent = label;
    btnRoute.disabled = !(origin && destination);
  }

  // ── Draw route on nav map ────────────────────────────────────
  function clearNavMap() {
    if (navRouteLayer) navRouteLayer.clearLayers();
  }

  function drawRouteOnNavMap(data) {
    initNavMap();
    clearNavMap();
    var coords = data.route_coordinates;
    if (!coords || coords.length < 2) return;

    var latlngs = coords.map(function (c) { return [c[0], c[1]]; });

    var polyline = L.polyline(latlngs, {
      color: '#2563eb',
      weight: 5,
      opacity: 0.85,
      lineJoin: 'round',
      lineCap: 'round'
    }).addTo(navRouteLayer);

    var oLat = data.origin.latitude, oLon = data.origin.longitude;
    var dLat = data.destination.actual_latitude || data.destination.latitude;
    var dLon = data.destination.actual_longitude || data.destination.longitude;

    var originName = origin ? origin.label : "Origin";
    var destName = destination ? destination.label : "Destination";

    L.marker([oLat, oLon], { icon: ORIGIN_ICON })
      .bindPopup('<div class="popup-label">Origin</div><div class="popup-name">' + escHtml(originName) + '</div>')
      .addTo(navRouteLayer);

    L.marker([dLat, dLon], { icon: DEST_ICON })
      .bindPopup('<div class="popup-label">Destination</div><div class="popup-name">' + escHtml(destName) + '</div>')
      .addTo(navRouteLayer);

    // Show blue user-position marker if GPS position is available
    if (lastGpsPosition && originType === "REAL") {
      createOrUpdateUserMarker(lastGpsPosition.lat, lastGpsPosition.lng, lastGpsPosition.accuracy);
    }

    var bounds = polyline.getBounds();
    if (bounds.isValid()) {
      var isMobile = window.innerWidth <= 600;
      var padVert = isMobile ? 60 : 80;
      var padHoriz = isMobile ? 30 : 50;
      navMap.fitBounds(bounds, { padding: [padVert, padHoriz], maxZoom: 16 });
    }

    setTimeout(function () { if (navMap) navMap.invalidateSize(); }, 150);
  }

  // ── Render route summary into nav bottom card ────────────────
  function renderNavResult(data) {
    var originLabel = origin ? origin.label : "Origin";
    var destLabel = destination ? destination.label : "Destination";
    var dist = data.distance_km;
    var eta = data.eta_minutes;
    var distStr = dist < 1 ? Math.round(dist * 1000) + " m" : dist.toFixed(2) + " km";
    var etaStr = eta < 1 ? "<1 min" : "~" + Math.round(eta) + " min";

    var h = "";
    h += '<div class="nav-route-found">Route found</div>';

    // Traffic source indicator
    var trafficSource = data.trafficSource || "historical_fallback";
    if (trafficSource === "tomtom_live") {
      h += '<div class="nav-traffic-badge nav-traffic-live">Live traffic data</div>';
    } else {
      h += '<div class="nav-traffic-badge nav-traffic-historical">Historical traffic data</div>';
    }

    h += '<div class="nav-endpoints">';
    h += '<div class="nav-ep"><span class="marker-dot" style="background:#16a34a"></span><span class="ep-name">' + escHtml(originLabel) + '</span></div>';
    h += '<div class="nav-connector"></div>';
    h += '<div class="nav-ep"><span class="marker-dot" style="background:#dc2626"></span><span class="ep-name">' + escHtml(destLabel) + '</span></div>';
    h += '</div>';
    h += '<div class="nav-metrics">';
    h += '<div><div class="metric-val">' + distStr + '</div><div class="metric-lbl">Distance</div></div>';
    h += '<div><div class="metric-val">' + etaStr + '</div><div class="metric-lbl">Estimated time</div></div>';
    h += '</div>';

    // Collapsible technical details
    h += '<button class="nav-details-toggle" data-action="toggle-nav-details">';
    h += '<span>Route details</span><span class="toggle-arrow">&#9660;</span>';
    h += '</button>';
    h += '<div class="nav-details-body" id="nav-details-body">';
    h += '<div class="nav-details-body-inner">';
    h += navDetailMetric("Origin type", originType === "DEMO" ? "DEMO (test coordinate)" : "REAL (browser GPS)");
    h += navDetailMetric("Origin snap", data.origin_snap_distance_m.toFixed(0) + " m");
    h += navDetailMetric("Destination snap", data.destination_snap_distance_m.toFixed(0) + " m");
    if (data.destinationSnapFallback) {
      h += '<div class="nav-detail-metric"><span class="dm-label">Snap mode</span><span class="dm-val" style="color:#f59e0b">Extended (400-600m)</span></div>';
    }
    h += navDetailMetric("Mean congestion ratio", data.mean_ratio.toFixed(4));
    h += navDetailMetric("Max congestion ratio", data.max_ratio.toFixed(4));
    h += navDetailMetric("Search time", data.search_time_seconds.toFixed(3) + " s");
    h += navDetailMetric("Request time", data.request_time_seconds.toFixed(3) + " s");
    h += navDetailMetric("Synthetic reverse edges", data.synthetic_reverse_edges);
    if (data.route && data.route.length > 0) {
      h += '<div class="nav-detail-segments">' + data.route.length + ' road segments in route</div>';
    }
    if (data.warnings && data.warnings.length > 0) {
      h += '<div class="nav-detail-warnings">';
      data.warnings.forEach(function (w) { h += escHtml(w) + "<br>"; });
      h += '</div>';
    }
    h += '</div></div>';

    navBottomCard.innerHTML = h;
  }

  function navDetailMetric(label, value) {
    return '<div class="nav-detail-metric"><span class="dm-label">' + escHtml(label) +
      '</span><span class="dm-val">' + escHtml(value) + '</span></div>';
  }

  // ── Geolocation ──────────────────────────────────────────────
  btnLocation.addEventListener("click", function () {
    if (!navigator.geolocation) {
      showStatus(originStatus, "Geolocation is not supported by your browser.", "error");
      return;
    }
    btnLocation.disabled = true;
    btnLocation.textContent = "Locating\u2026";
    hideStatus(originStatus);
    stopGpsTracking();

    navigator.geolocation.getCurrentPosition(
      function (pos) {
        var lat = pos.coords.latitude;
        var lon = pos.coords.longitude;
        setOrigin(lat, lon, lat.toFixed(4) + ", " + lon.toFixed(4), "REAL");
        showStatus(originStatus, "Location acquired: " + lat.toFixed(5) + ", " + lon.toFixed(5), "success");
        originHint.classList.add("hidden");
        btnLocation.disabled = false;
        btnLocation.textContent = "Use My Location";
        startGpsTracking();
      },
      function (err) {
        var msg;
        switch (err.code) {
          case err.PERMISSION_DENIED:
            msg = "Location permission denied. Please allow location access or use Delhi Demo Location.";
            break;
          case err.POSITION_UNAVAILABLE:
            msg = "Location information is unavailable.";
            break;
          case err.TIMEOUT:
            msg = "Location request timed out. Please try again.";
            break;
          default:
            msg = "An unknown error occurred while retrieving location.";
        }
        showStatus(originStatus, msg, "error");
        btnLocation.disabled = false;
        btnLocation.textContent = "Use My Location";
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 }
    );
  });

  // ── Delhi demo location ──────────────────────────────────────
  var DELHI_DEMO = { lat: 28.5692, lon: 77.2090, label: "Safdarjung, Delhi (demo)" };

  btnDemo.addEventListener("click", function () {
    stopGpsTracking();
    setOrigin(DELHI_DEMO.lat, DELHI_DEMO.lon, DELHI_DEMO.label, "DEMO");
    showStatus(
      originStatus,
      "Demo origin: Safdarjung (" + DELHI_DEMO.lat + ", " + DELHI_DEMO.lon + ") \u2014 test coordinate, not your real location.",
      "info"
    );
    originHint.classList.add("hidden");
  });

  // ── Destination search ───────────────────────────────────────
  searchInput.addEventListener("input", function () {
    var q = searchInput.value.trim();
    clearTimeout(searchTimeout);
    if (q.length < 2) {
      resultsList.classList.add("hidden");
      resultsList.innerHTML = "";
      hideStatus(destStatus);
      return;
    }
    searchTimeout = setTimeout(function () { doSearch(q); }, 300);
  });

  async function doSearch(q) {
    searchSpinner.classList.remove("hidden");
    resultsList.classList.add("hidden");
    resultsList.innerHTML = "";
    hideStatus(destStatus);

    var url = "/api/places?q=" + encodeURIComponent(q) + "&limit=5";
    if (origin) { url += "&lat=" + origin.lat + "&lon=" + origin.lon; }

    try {
      var resp = await fetch(url);
      if (!resp.ok) {
        var body = await resp.json().catch(function () { return {}; });
        throw new Error(body.detail || "Server returned " + resp.status);
      }
      var data = await resp.json();
      searchSpinner.classList.add("hidden");

      if (!data.results || data.results.length === 0) {
        resultsList.innerHTML = '<div class="empty">No results for "' + escHtml(q) + '"</div>';
        resultsList.classList.remove("hidden");
        return;
      }
      renderResults(data.results);
    } catch (e) {
      searchSpinner.classList.add("hidden");
      showStatus(destStatus, "Search failed: " + e.message, "error");
    }
  }

  function renderResults(results) {
    resultsList.innerHTML = "";
    results.forEach(function (r) {
      var div = document.createElement("div");
      div.className = "result-item";
      div.innerHTML =
        '<div class="name">' + escHtml(r.name || "Unnamed") + "</div>" +
        '<div class="detail">' + escHtml(r.display_name || "") + "</div>";
      div.addEventListener("click", function () {
        setDestination(r.latitude, r.longitude, r.name || "Unnamed", r.display_name);
        resultsList.classList.add("hidden");
        resultsList.innerHTML = "";
        searchInput.value = r.name || "Unnamed";
        hideStatus(destStatus);
        showStatus(destStatus, "Selected: " + (r.name || "Unnamed"), "success");
      });
      resultsList.appendChild(div);
    });
    resultsList.classList.remove("hidden");
  }

  // ── Find route ───────────────────────────────────────────────
  btnRoute.addEventListener("click", function () {
    if (!origin || !destination) return;

    var dt = datetimeInput.value;
    if (!dt) {
      showStatus(routeStatus, "Please select a departure date/time.", "error");
      return;
    }

    var departureStr = dt.replace("T", " ").replace(/:\d{2}$/, "") + ":00";

    var payload = {
      origin_lat: origin.lat,
      origin_lon: origin.lon,
      dest_lat: destination.lat,
      dest_lon: destination.lon,
      departure_datetime: departureStr,
    };

    btnRoute.disabled = true;
    btnRoute.textContent = "Computing\u2026";
    hideStatus(routeStatus);
    navBottomCard.innerHTML = "";

    showStatus(routeStatus, "Computing route\u2026", "info");

    fetch("/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (resp) {
        return resp.json().then(function (body) {
          return { status: resp.status, body: body };
        });
      })
      .then(function (res) {
        btnRoute.disabled = false;
        btnRoute.textContent = "Find Route";
        hideStatus(routeStatus);

        if (res.status !== 200) {
          var detail = res.body.detail || "Unknown error";
          var friendly = humaniseRouteError(detail, res.status);
          showStatus(routeStatus, friendly, "error");
          logFieldTest({ success: false, error: detail });
          return;
        }

        lastRouteData = res.body;
        drawRouteOnNavMap(res.body);
        renderNavResult(res.body);
        showNavigation();
        logFieldTest({
          success: true,
          originSnap: res.body.origin_snap_distance_m,
          destSnap: res.body.destination_snap_distance_m,
          distanceKm: res.body.distance_km,
          etaMin: res.body.eta_minutes,
          warnings: res.body.warnings || [],
          trafficSource: res.body.trafficSource || null,
          trafficDataAvailable: res.body.trafficDataAvailable || false,
          trafficIncidentCount: res.body.trafficIncidentCount || 0,
        });
      })
      .catch(function (e) {
        btnRoute.disabled = false;
        btnRoute.textContent = "Find Route";
        showStatus(routeStatus, "Network error: " + e.message, "error");
        logFieldTest({ success: false, error: "network: " + e.message });
      });
  });

  // ── Back button ──────────────────────────────────────────────
  btnBack.addEventListener("click", function () {
    showPlanning();
  });

  // ── Nav bottom card delegated events ─────────────────────────
  navBottomCard.addEventListener("click", function (e) {
    var target = e.target.closest("[data-action]");
    if (!target) return;
    if (target.getAttribute("data-action") === "toggle-nav-details") {
      var body = navBottomCard.querySelector("#nav-details-body");
      if (body) {
        body.classList.toggle("open");
        target.classList.toggle("open");
      }
    }
  });

  // ── Error humanisation ───────────────────────────────────────
  function humaniseRouteError(detail, status) {
    if (typeof detail !== "string") detail = JSON.stringify(detail);

    if (detail.indexOf("snap radius") !== -1) {
      var isOrigin = detail.indexOf("Origin:") === 0;
      var isDest = detail.indexOf("Destination:") === 0;

      if (isDest) {
        return "The destination is outside the supported Delhi/NCR area. Try a destination closer to a major road within Delhi, or use the Delhi Demo Location as your origin for testing.";
      }
      if (isOrigin && originType === "DEMO") {
        return "The demo origin could not be matched to the road network. This is unexpected — please try again.";
      }
      return "PredictRoute currently supports Delhi/NCR only. Try a location within Delhi/NCR, or use Delhi Demo Location for testing.";
    }
    if (detail.indexOf("disconnected") !== -1 || detail.indexOf("strongly connected") !== -1) {
      var isOriginDC = detail.indexOf("Origin:") === 0;
      if (isOriginDC) {
        return "The origin is in a disconnected area of the road network and cannot reach the destination.";
      }
      return "The destination is in an unreachable area of the road network.";
    }
    if (detail.indexOf("Routing error") !== -1 || detail.indexOf("UNREACHABLE") !== -1) {
      return "No route could be found between the selected origin and destination. They may be in disconnected parts of the road network.";
    }
    return "An error occurred while computing the route. Please try again.";
  }

  // ── Field-test logging ───────────────────────────────────────
  var LOG_KEY = "predictroute_field_test_log";

  function logFieldTest(result) {
    var entry = {
      ts: new Date().toISOString(),
      originType: originType || "UNKNOWN",
      destLabel: destination ? destination.label : null,
      success: result.success,
      originSnapM: result.originSnap != null ? Math.round(result.originSnap) : null,
      destSnapM: result.destSnap != null ? Math.round(result.destSnap) : null,
      distanceKm: result.distanceKm != null ? +result.distanceKm.toFixed(2) : null,
      etaMin: result.etaMin != null ? +result.etaMin.toFixed(1) : null,
      warnings: result.warnings || [],
      error: result.error || null,
      liveTrackingStarted: result.liveTrackingStarted || null,
      liveTrackingError: result.liveTrackingError || null,
      liveTrackingStopped: result.liveTrackingStopped || null,
      trafficSource: result.trafficSource || null,
      trafficDataAvailable: result.trafficDataAvailable != null ? result.trafficDataAvailable : null,
      trafficIncidentCount: result.trafficIncidentCount != null ? result.trafficIncidentCount : null,
    };
    try {
      var log = JSON.parse(localStorage.getItem(LOG_KEY) || "[]");
      log.push(entry);
      localStorage.setItem(LOG_KEY, JSON.stringify(log));
    } catch (e) { /* localStorage unavailable */ }
  }

  window.predictRouteExportLog = function () {
    try { return JSON.parse(localStorage.getItem(LOG_KEY) || "[]"); }
    catch (e) { return []; }
  };
  window.predictRouteClearLog = function () { localStorage.removeItem(LOG_KEY); };

  // ── Footer buttons ───────────────────────────────────────────
  $("#btn-export-log").addEventListener("click", function () {
    var l = window.predictRouteExportLog();
    if (!l.length) { alert("No test log entries yet."); return; }
    var blob = new Blob([JSON.stringify(l, null, 2)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "predictroute_field_test_log.json";
    a.click();
  });

  $("#btn-clear-log").addEventListener("click", function () {
    if (confirm("Clear all test log entries?")) {
      window.predictRouteClearLog();
      alert("Log cleared.");
    }
  });

  // ── Cleanup GPS tracking on page unload ──────────────────────
  window.addEventListener("beforeunload", function () {
    stopGpsTracking();
  });
})();
