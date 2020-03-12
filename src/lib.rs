use Debug;
use serde::{Deserialize, Serialize};
use chrono::prelude::*;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, PartialOrd)]
pub struct DataVector<T: Clone> {
    pub name: String,
    pub x_units: String,
    pub x_name: String,
    pub y_units: String,
    pub y_name: String,
    /// The values associated with this vector. TODO: stop allowing public
    /// access.
    pub values: Vec<Point<T>>,
}

// We need an interpolatable trait. This includes this like dates.
pub trait Interpolate {
    fn interpolate(x:f64, x1: f64, x2: f64, y1: Self, y2: Self) -> Self;
}

impl Interpolate for f64 {
    fn interpolate(x:f64, x1: f64, x2: f64, y1: Self, y2: Self) -> Self {
        ((x - x1) / (x2 - x1)) * (y2 - y1) + y1
    }
}

impl<Tz: TimeZone> Interpolate for chrono::DateTime<Tz> {
    fn interpolate(x:f64, x1: f64, x2: f64, y1: Self, y2: Self) -> Self {
        let basis = 1000 as f64;
        let y_diff = y2 - y1.clone();
        y1 + (y_diff*(((x - x1)*basis) as i32) / (((x2 - x1)*basis) as i32))/(basis as i32)
    }
}

// pub

impl<T: Clone + PartialOrd + Interpolate> DataVector<T> {
    /// A getter method for the values. There is no field access as we don't
    /// want to allow arbitrary changing that might result in unordered data.
    pub fn values(&self) -> &Vec<Point<T>> {
        &self.values
    }

    pub fn combined_iter<'a>(&'a self, other: &'a DataVector<T>) -> CombinedDVIter<T> {
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
    pub fn resample_max(&self, other: &DataVector<T>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: max_or_first(p1.y, p2.y),
                },
                WhichVector::First(p) => {
                    let y = match other.interpolate(p.x) {
                        Some(second_y) => max_or_first(p.y, second_y),
                        None => p.y,
                    };
                    Point { x: p.x, y }
                }
                WhichVector::Second(p) => {
                    let y = match self.interpolate(p.x) {
                        Some(first_y) => max_or_first(p.y, first_y),
                        None => p.y,
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
}

impl<T: Interpolate + PartialOrd + Zero + One + Clone + core::ops::Add<T, Output = T> + core::ops::Div<T, Output = T>> DataVector<T> {
    /// Resample as an average of two vectors.
    pub fn resample_avg(&self, other: &DataVector<T>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: (p1.y + p2.y) / (T::one() + T::one()),
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(T::zero());
                    Point {
                        x: p.x,
                        y: (p.y + y) / (T::one() + T::one()),
                    }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(T::zero());
                    Point {
                        x: p.x,
                        y: (p.y + y) / (T::one() + T::one()),
                    }
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

impl<T: Clone + Interpolate> DataVector<T> {

    fn interpolate(&self, x: f64) -> Option<T> {
        if self.values.len() == 0 {
            return None;
        }
        // We assume that the values are properly sorted on the x-axis.
        for i in 0..self.values.len() {
            let this_point = self.values[i].clone();
            if x < this_point.x {
                return None;
            }
            if let Some(next_point) = self.values.get(i + 1) {
                if x > next_point.x {
                    continue;
                } else {
                    let x1 = this_point.x;
                    let x2 = next_point.x;
                    let y1 = this_point.y;
                    let y2 = next_point.y.clone();
                    // Value is between this_point and next_point.
                    return Some(T::interpolate(x, x1, x2, y1, y2));
                }
            } else {
                return None;
            }
        }
        None
    }
}

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for std::time::Duration {
    fn zero() -> Self {
        std::time::Duration::from_secs(0)
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl<T> DataVector<T>
    where
    T: Clone + Zero + PartialOrd + Interpolate + core::ops::Add<T, Output = T> + core::ops::Sub,
{
    /// Resample self onto another vector. That is, interpolate to create a new
    /// vector with the same x-axis as ['other']. Actually, we need all the
    /// points of both to preserve accuracy.
    pub fn resample_add(&self, other: &DataVector<T>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point: Point<T> = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: p1.y + p2.y,
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(T::zero());
                    Point { x: p.x, y: p.y + y }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(T::zero());
                    Point { x: p.x, y: p.y + y }
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

fn max_or_first<T: PartialOrd>(a: T, b: T) -> T {
    if b < a {
        a
    } else {
        b
    }
}

pub struct CombinedDVIter<'a,T> {
    first_values: &'a Vec<Point<T>>,
    second_values: &'a Vec<Point<T>>,
    next_first_i: usize,
    next_second_i: usize,
}

impl<'a,T: Clone> Iterator for CombinedDVIter<'a,T> {
    type Item = WhichVector<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // See what points are next.
        let next_first_point_opt = self.first_values.get(self.next_first_i);
        let next_second_point_opt = self.second_values.get(self.next_second_i);
        match (next_first_point_opt, next_second_point_opt) {
            (Some(first_point), None) => {
                self.next_first_i += 1;
                Some(WhichVector::First((*first_point).clone()))
            }
            (None, Some(second_point)) => {
                self.next_second_i += 1;
                Some(WhichVector::Second((*second_point).clone()))
            }
            (Some(first_point), Some(second_point)) => {
                if first_point.x < second_point.x {
                    self.next_first_i += 1;
                    Some(WhichVector::First((*first_point).clone()))
                } else if second_point.x < first_point.x {
                    self.next_second_i += 1;
                    Some(WhichVector::Second((*second_point).clone()))
                } else if first_point.x == second_point.x {
                    self.next_first_i += 1;
                    self.next_second_i += 1;
                    Some(WhichVector::Both((*first_point).clone(), (*second_point).clone()))
                } else {
                    panic!("invalide equality test");
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WhichVector<T> {
    /// We have a value from the first vector but not the second.
    First(Point<T>),
    /// We have a value from the seconds vector but not the first.
    Second(Point<T>),
    /// We have values from both vectors.
    Both(Point<T>, Point<T>),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Point<T> {
    pub x: f64,
    pub y: T,
}

use std::cmp::Ordering;
impl<T: PartialOrd> Ord for Point<T> {
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

impl<T: PartialEq> Eq for Point<T> {}

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
        let ci: Vec<WhichVector<f64>> = dv1.combined_iter(&dv2).collect();
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
                Point { x: 0_f64, y: 0_f64 },
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
                Point { x: 0_f64, y: 0_f64 },
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
