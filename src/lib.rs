use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct DataVector {
    pub name: String,
    pub x_units: String,
    pub x_name: String,
    pub y_units: String,
    pub y_name: String,
    /// The values associated with this vector. TODO: stop allowing public
    /// access.
    pub values: Vec<Point>,
}

impl DataVector {
    /// A getter method for the values. There is no field access as we don't
    /// want to allow arbitrary changing that might result in unordered data.
    pub fn values(&self) -> &Vec<Point> {
        &self.values
    }

    /// Given an x value, linearly interpolate to find a y value. Return none if
    /// it is out of bounds.
    pub fn interpolate(&self, x: f64) -> Option<f64> {
        if self.values.len() == 0 {
            return None;
        }
        // We assume that the values are properly sorted on the x-axis.
        for i in 0..self.values.len() {
            let this_point = self.values[i];
            if x < this_point.x {
                return None;
            }
            if let Some(next_point) = self.values.get(i+1) {
                if x > next_point.x {
                    continue;
                } else {
                    let x1 = this_point.x;
                    let x2 = next_point.x;
                    let y1 = this_point.y;
                    let y2 = next_point.y;
                    // Value is between this_point and next_point.
                    return Some(((x - x1) / (x2 - x1)) * (y2 - y1) + y1);
                }
            } else {
                return None;
            }
        }
        None
    }

    pub fn combined_iter<'a>(&'a self, other: &'a DataVector) -> CombinedDVIter {
        CombinedDVIter {
            first_values: &self.values,
            second_values: &other.values,
            next_first_i: 0,
            next_second_i: 0,
        }
    }

    /// Resample self onto another vector. That is, interpolate to create a new
    /// vector with the same x-axis as ['other']. Actually, we need all the
    /// points of both to preserve accuracy.
    pub fn resample_add(&self, other: &DataVector, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: p1.y + p2.y,
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(0.0);
                    Point { x: p.x, y:p.y+y }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(0.0);
                    Point { x: p.x, y:p.y+y }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }

    /// Resample self onto another vector. That is, interpolate to create a new
    /// vector with the same x-axis as ['other']. Actually, we need all the
    /// points of both to preserve accuracy.
    pub fn resample_max(&self, other: &DataVector, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: max_f64(p1.y, p2.y),
                },
                WhichVector::First(p) => {
                    let y = match other.interpolate(p.x) {
                        Some(second_y) => max_f64(p.y, second_y),
                        None => p.y
                    };
                    Point { x: p.x, y }
                }
                WhichVector::Second(p) => {
                    let y = match self.interpolate(p.x) {
                        Some(first_y) => max_f64(p.y, first_y),
                        None => p.y
                    };
                    Point { x: p.x, y }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }


    /// Resample as an average of two vectors.
    pub fn resample_avg(&self, other: &DataVector, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: (p1.y + p2.y)/2.0,
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(0.0);
                    Point { x: p.x, y:(p.y+y)/2.0 }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(0.0);
                    Point { x: p.x, y:(p.y+y)/2.0 }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}

fn cmp_f64(a:f64,b:f64) -> std::cmp::Ordering {
    if a < b {
        Ordering::Less
    } else if a == b {
        Ordering::Equal
    } else {
        Ordering::Greater
    }
}

fn max_f64(a:f64,b:f64) -> f64 {
    match cmp_f64(a, b) {
        Ordering::Greater => a,
        _ => b,
    }
}

pub struct CombinedDVIter<'a> {
    first_values: &'a Vec<Point>,
    second_values: &'a Vec<Point>,
    next_first_i: usize,
    next_second_i: usize,
}

impl<'a> Iterator for CombinedDVIter<'a> {
    type Item = WhichVector;

    fn next(&mut self) -> Option<Self::Item> {
        // See what points are next.
        let next_first_point_opt = self.first_values.get(self.next_first_i);
        let next_second_point_opt = self.second_values.get(self.next_second_i);
        match (next_first_point_opt, next_second_point_opt) {
            (Some(first_point), None) => {
                self.next_first_i += 1;
                Some(WhichVector::First(*first_point))
            }
            (None, Some(second_point)) => {
                self.next_second_i += 1;
                Some(WhichVector::Second(*second_point))
            }
            (Some(first_point), Some(second_point)) => {
                if first_point.x < second_point.x {
                    self.next_first_i += 1;
                    Some(WhichVector::First(*first_point))
                } else if second_point.x < first_point.x {
                    self.next_second_i += 1;
                    Some(WhichVector::Second(*second_point))
                } else if first_point.x == second_point.x {
                    self.next_first_i += 1;
                    self.next_second_i += 1;
                    Some(WhichVector::Both(*first_point, *second_point))
                } else {
                    panic!("invalide equality test");
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WhichVector {
    /// We have a value from the first vector but not the second.
    First(Point),
    /// We have a value from the seconds vector but not the first.
    Second(Point),
    /// We have values from both vectors.
    Both(Point, Point),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

use std::cmp::Ordering;
impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.x < other.x {
            Ordering::Less
        } else if self.x == other.x {
            if self.y < other.y {
                Ordering::Less
            } else if self.y == other.y {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        } else {
            Ordering::Greater
        }
    }
}

impl Eq for Point {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation() {
        let dv = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        assert_eq!(Some(50.0), dv.interpolate(50.0));
        assert_eq!(Some(75.0), dv.interpolate(75.0));
    }

    #[test]
    fn iters() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci: Vec<WhichVector> = dv1.combined_iter(&dv2).collect();
        eprintln!("{:?}", ci);
    }

    #[test]
    fn add_vectors() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv3 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 0_f64,
                    y: 0_f64,
                },
                Point {
                    x: 10_f64,
                    y: 10_f64,
                },
                Point {
                    x: 70_f64,
                    y: 170_f64,
                },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci = dv1.resample_add(&dv2, "Test DV".to_string());
        eprintln!("{:?}", ci);
        assert_eq!(dv3, ci);
    }


    #[test]
    fn max_vectors() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv3 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 0_f64,
                    y: 0_f64,
                },
                Point {
                    x: 10_f64,
                    y: 10_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci = dv1.resample_max(&dv2, "Test DV".to_string());
        eprintln!("{:?}", ci);
        assert_eq!(dv3, ci);
    }

}
