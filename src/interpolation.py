import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

def interpolate_cable_linear(points):
    if len(points) < 2:
        return points
    pca = PCA(n_components=1)
    proj = pca.fit_transform(points)
    sorted_indices = np.argsort(proj.flatten())
    sorted_points = points[sorted_indices]
    
    # Tri explicite selon laxe Y pour éviter les zigzags
    sorted_points = sorted_points[np.argsort(sorted_points[:,1])]
    
    t = np.linspace(0, 1, len(sorted_points))
    interpolator = interp1d(t, sorted_points, axis=0, kind='linear', bounds_error=False)
    t_interp = np.linspace(0, 1, max(10*len(points), 50))
    interpolated = interpolator(t_interp)
    return interpolated

def interpolate_cable_catenary(points):
    if len(points) < 3:
        return interpolate_cable_linear(points)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(points)
    sorted_indices = np.argsort(proj.flatten())
    sorted_points = points[sorted_indices]
    
    # Tri explicite selon laxe Y pour éviter les zigzags
    sorted_points = sorted_points[np.argsort(sorted_points[:,1])]
    
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    z = sorted_points[:, 2]
    try:
        from scipy.optimize import curve_fit
        def catenary_model(y, a, y0, b):
            return a * np.cosh((y - y0) / a) + b
        popt, _ = curve_fit(catenary_model, y, z, p0=[10, np.mean(y), np.mean(z)], maxfev=1000)
        y_interp = np.linspace(y.min(), y.max(), max(10*len(points), 50))
        z_interp = catenary_model(y_interp, *popt)
        x_interp = np.linspace(x.min(), x.max(), len(y_interp))
        interpolated = np.column_stack([x_interp, y_interp, z_interp])
    except Exception:
        interpolated = interpolate_cable_linear(points)
    return interpolated

def interpolate_all_cables(points, labels, method='linear'):
    unique_labels = np.unique(labels[labels != -1])
    interpolated_cables = {}
    for label in unique_labels:
        cluster_points = points[labels == label]
        if method == 'linear':
            interpolated = interpolate_cable_linear(cluster_points)
        elif method == 'catenary':
            interpolated = interpolate_cable_catenary(cluster_points)
        else:
            interpolated = interpolate_cable_linear(cluster_points)
        interpolated_cables[label] = interpolated
    return interpolated_cables

def generate_catenary_curve(start_point, end_point, num_points=50, sag=0.1):
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([start_point])
    t = np.linspace(0, 1, num_points)
    x = start_point[0] + t * direction[0]
    y = start_point[1] + t * direction[1]
    z = start_point[2] + t * direction[2] + sag * length * (t - t**2)
    return np.column_stack([x, y, z]) 